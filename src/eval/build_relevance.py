"""
src/eval/build_relevance.py
---------------------------
One-time helper that:
  1. Loads Flickr8k captions from Hugging Face cache (intro/flickr8k).
  2. Matches test queries against captions using keyword rules.
  3. Filters matches to only include images currently indexed in models/image_paths.json.
  4. Writes src/eval/test_queries.json with REAL, caption-verified relevance labels.

Run once before evaluate_retrieval.py:
    python -m src.eval.build_relevance
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Test queries + keyword rules
# Each rule is a list of keyword groups; a caption must contain at least one
# keyword from EVERY group (AND logic between groups, OR within a group).
# ---------------------------------------------------------------------------
QUERY_RULES = [
    {
        "query": "a dog running outdoors",
        "groups": [["dog", "dogs"], ["run", "running", "races", "sprints", "chasing", "grass", "field"]],
    },
    {
        "query": "children playing in water",
        "groups": [["child", "children", "boy", "boys", "girl", "girls", "kid", "kids"],
                   ["water", "pool", "ocean", "river", "lake", "splash", "swim", "waves", "beach"]],
    },
    {
        "query": "a person riding a bicycle",
        "groups": [["bike", "bicycle", "cycling", "cyclist", "bicyclist", "rides a bicycle", "riding a bike"]],
    },
    {
        "query": "two dogs playing together",
        "groups": [["two dogs", "dogs"], ["play", "playing", "together", "wrestling", "chasing"]],
    },
    {
        "query": "a man jumping in the air",
        "groups": [["man", "boy", "person", "player"],
                   ["jump", "jumps", "jumping", "leaps", "leaping", "midair", "air"]],
    },
    {
        "query": "a group of people outdoors",
        "groups": [["group", "crowd", "people", "children", "men", "women", "players"],
                   ["outdoor", "outdoors", "outside", "field", "park", "street", "beach"]],
    },
    {
        "query": "a dog swimming in water",
        "groups": [["dog", "dogs"], ["swim", "swimming", "water", "lake", "river", "pool"]],
    },
    {
        "query": "a person wearing a red shirt",
        "groups": [["man", "woman", "person", "boy", "girl", "child", "lady"],
                   ["red shirt", "red jacket", "red top", "red sweater", "red coat", "red jersey"]],
    },
    {
        "query": "a dog running on grass",
        "groups": [["dog", "dogs"], ["run", "running"], ["grass", "lawn", "field", "yard", "green"]],
    },
    {
        "query": "a smiling young child",
        "groups": [["boy", "child", "kid", "girl", "baby", "toddler"],
                   ["smile", "smiles", "smiling", "laugh", "laughing", "happy", "grin"]],
    },
    {
        "query": "people playing soccer on a field",
        "groups": [["soccer", "football", "match"],
                   ["player", "players", "field", "game", "ball"]],
    },
    {
        "query": "a person hiking in the mountains",
        "groups": [["hik", "climb", "mountain", "trail", "rocky", "rocks", "backpack", "summit"]],
    },
    {
        "query": "a black dog playing with a ball",
        "groups": [["black dog", "dark dog"],
                   ["ball", "toy", "fetch", "catch", "play", "mouth"]],
    },
    {
        "query": "people sitting on a bench outdoors",
        "groups": [["bench", "sitting on a bench", "sits on a bench"]],
    },
    {
        "query": "a bride and groom at a wedding",
        "groups": [["bride", "groom", "wedding", "tuxedo"]],
    },
]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_PATHS_FILE = BASE_DIR / "models" / "image_paths.json"
EVAL_DIR = Path(__file__).parent
OUT_FILE = EVAL_DIR / "test_queries.json"

MIN_IMAGES = 3   # skip query if fewer matches found
MAX_IMAGES = 25  # keep a solid pool of relevant items for evaluation


def caption_matches(caption: str, groups: list) -> bool:
    """Return True if caption satisfies all keyword groups (AND logic)."""
    caption_lower = caption.lower()
    for group in groups:
        if not any(kw in caption_lower for kw in group):
            return False
    return True


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Run: pip install datasets\n"
            "Then re-run: python -m src.eval.build_relevance"
        )

    # 1. Load indexed images so we only evaluate against images actually in the model index
    indexed_filenames = set()
    if IMAGE_PATHS_FILE.exists():
        with open(IMAGE_PATHS_FILE, "r", encoding="utf-8") as fh:
            paths = json.load(fh)
            indexed_filenames = {Path(p).name for p in paths}
        print(f"Loaded {len(indexed_filenames)} indexed image filenames from {IMAGE_PATHS_FILE.name}")
    else:
        print(f"Warning: {IMAGE_PATHS_FILE} not found. Using data/images directly.")
        images_dir = BASE_DIR / "data" / "images"
        if images_dir.exists():
            indexed_filenames = {p.name for p in images_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    print("Loading Flickr8k captions from Hugging Face...")
    ds_dict = load_dataset("intro/flickr8k")
    print(f"  Dataset splits: {list(ds_dict.keys())}")

    # Build mapping: filename -> list of captions
    caption_map: dict[str, list[str]] = {}

    for split_name, ds in ds_dict.items():
        for row in ds:
            # Extract image filename
            fname = None
            img_obj = row.get("image")
            if hasattr(img_obj, "filename") and img_obj.filename:
                fname = Path(str(img_obj.filename).replace("\\", "/")).name
            elif "file_name" in row:
                fname = Path(row["file_name"]).name

            if not fname:
                continue

            # Only retain if the image exists in our indexed database
            if indexed_filenames and fname not in indexed_filenames:
                continue

            # Extract all 5 captions
            captions = []
            for i in range(5):
                key = f"caption_{i}"
                if key in row and row[key]:
                    captions.append(str(row[key]))
            if not captions and "caption" in row and row["caption"]:
                captions.append(str(row["caption"]))

            if captions:
                caption_map.setdefault(fname, []).extend(captions)

    print(f"  Total indexed images with captions: {len(caption_map)}\n")

    queries_out = []
    for rule in QUERY_RULES:
        query = rule["query"]
        groups = rule["groups"]
        min_img = rule.get("min_images", MIN_IMAGES)

        matched = []
        for fname, captions in caption_map.items():
            if any(caption_matches(cap, groups) for cap in captions):
                matched.append(f"data/images/{fname}")

        matched = matched[:MAX_IMAGES]

        if len(matched) < min_img:
            print(f"  [SKIP] \"{query}\" - only {len(matched)} matches (need >= {min_img})")
            continue

        print(f"  [OK]   \"{query}\" -> {len(matched)} relevant images")
        queries_out.append({"query": query, "relevant": matched})

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(queries_out, fh, indent=2)

    print(f"\nSaved {len(queries_out)} validated queries to: {OUT_FILE}")


if __name__ == "__main__":
    main()
