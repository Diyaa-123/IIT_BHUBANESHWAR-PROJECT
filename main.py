import argparse

from src.engine import ImageSearchEngine


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Image Search Engine using CLIP"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language search query"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of images to retrieve"
    )

    args = parser.parse_args()

    try:
        engine = ImageSearchEngine()

        results = engine.search(
            query=args.query,
            top_k=args.top_k
        )

        print(f"\nQuery: {args.query}")
        print(f"\nTop {len(results)} results:\n")

        for rank, (path, score) in enumerate(results, start=1):
            print(f"{rank}. {path}")
            print(f"   Similarity: {score:.4f}")

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
