"""
app.py — Week 7 Streamlit Frontend
AI-Powered Image Search Engine with Griffin-inspired premium UI design.
"""

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    EMBEDDINGS_FILE,
    IMAGE_PATHS_FILE,
    DEFAULT_TOP_K,
    CLIP_MODEL_NAME,
    DEVICE,
)
from src.embeddings import CLIPEmbedder
from src.search import top_k_search


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visora — AI Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Griffin-inspired CSS ───────────────────────────────────────────────────────
GRIFFIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --color-bg:         #0d0d0e;
    --color-surface:    #141416;
    --color-border:     #232328;
    --color-border-h:   #3a3a42;
    --color-text:       #f0ece4;
    --color-muted:      #8a8a96;
    --color-dimmed:     #4a4a54;
    --color-accent:     #c8f07a;
    --color-accent-dim: rgba(200,240,122,0.08);
    --color-accent-glow:rgba(200,240,122,0.15);
    --color-tag-bg:     #1e1e22;
    --color-pill-bg:    rgba(200,240,122,0.1);
    --font-serif: 'DM Serif Display', Georgia, serif;
    --font-sans:  'DM Sans', system-ui, sans-serif;
    --font-mono:  'DM Mono', monospace;
    --r-sm: 6px; --r-md: 12px;
    --t: 0.2s ease;
}

*{box-sizing:border-box;}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,section.main{
    background-color:var(--color-bg)!important;
    color:var(--color-text)!important;
    font-family:var(--font-sans)!important;
}

[data-testid="stHeader"]{background:var(--color-bg)!important;border-bottom:1px solid var(--color-border)!important;}
[data-testid="stSidebar"]{background:var(--color-surface)!important;border-right:1px solid var(--color-border)!important;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}

::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--color-bg);}
::-webkit-scrollbar-thumb{background:var(--color-border-h);border-radius:99px;}

.nav-bar{
    display:flex;align-items:center;justify-content:space-between;
    padding:20px 48px;border-bottom:1px solid var(--color-border);
    background:rgba(13,13,14,0.95);position:sticky;top:0;z-index:100;
    backdrop-filter:blur(12px);
}
.nav-logo{display:flex;align-items:center;gap:10px;}
.nav-logo-mark{
    width:32px;height:32px;background:var(--color-accent);border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-family:var(--font-mono);font-size:14px;font-weight:500;
    color:#0d0d0e;letter-spacing:-0.5px;flex-shrink:0;
}
.nav-logo-text{font-family:var(--font-serif);font-size:20px;color:var(--color-text);letter-spacing:-0.03em;}
.nav-pill{
    font-family:var(--font-mono);font-size:11px;font-weight:500;text-transform:uppercase;
    letter-spacing:0.1em;color:var(--color-accent);background:var(--color-pill-bg);
    border:1px solid rgba(200,240,122,0.2);padding:4px 12px;border-radius:99px;
}

.hero-section{padding:72px 48px 48px;max-width:880px;margin:0 auto;text-align:center;}
.hero-eyebrow{
    font-family:var(--font-mono);font-size:11px;font-weight:500;text-transform:uppercase;
    letter-spacing:0.14em;color:var(--color-muted);margin-bottom:20px;
    display:flex;align-items:center;justify-content:center;gap:8px;
}
.hero-eyebrow::before,.hero-eyebrow::after{
    content:'';display:block;height:1px;width:32px;background:var(--color-border-h);
}
.hero-title{
    font-family:var(--font-serif);font-size:clamp(42px,6vw,72px);font-weight:400;
    letter-spacing:-0.04em;line-height:1.08;color:var(--color-text);margin:0 0 20px;
}
.hero-title em{font-style:italic;color:var(--color-accent);}
.hero-sub{
    font-size:17px;font-weight:300;line-height:1.6;color:var(--color-muted);
    margin:0 auto 48px;max-width:560px;
}

[data-testid="stTextInput"] label{display:none!important;}
[data-testid="stTextInput"] input{
    background:var(--color-surface)!important;border:1px solid var(--color-border)!important;
    border-radius:var(--r-md)!important;color:var(--color-text)!important;
    font-family:var(--font-sans)!important;font-size:16px!important;font-weight:300!important;
    padding:16px 20px!important;height:56px!important;
    transition:border-color var(--t),box-shadow var(--t)!important;
    caret-color:var(--color-accent)!important;
}
[data-testid="stTextInput"] input:focus{
    border-color:rgba(200,240,122,0.4)!important;
    box-shadow:0 0 0 3px var(--color-accent-glow)!important;outline:none!important;
}
[data-testid="stTextInput"] input::placeholder{color:var(--color-dimmed)!important;font-style:italic;}

[data-testid="stButton"]>button{
    font-family:var(--font-mono)!important;font-size:13px!important;font-weight:500!important;
    text-transform:uppercase!important;letter-spacing:0.06em!important;
    border-radius:var(--r-sm)!important;transition:all var(--t)!important;
    height:42px!important;padding:0 24px!important;cursor:pointer!important;
}
[data-testid="stButton"][data-type="primaryFormSubmit"]>button,
div[data-testid="column"]:first-child [data-testid="stButton"]>button,
[data-testid="stFormSubmitButton"]:first-of-type>button{
    background:var(--color-accent)!important;color:#0d0d0e!important;border:none!important;
}
[data-testid="stFormSubmitButton"]>button{
    background:var(--color-accent)!important;color:#0d0d0e!important;border:none!important;
}
[data-testid="stFormSubmitButton"]>button:hover{
    background:#d4f58a!important;transform:translateY(-1px)!important;
    box-shadow:0 4px 20px var(--color-accent-glow)!important;
}
[data-testid="stFormSubmitButton"]:last-child>button{
    background:transparent!important;color:var(--color-muted)!important;
    border:1px solid var(--color-border)!important;
}
[data-testid="stFormSubmitButton"]:last-child>button:hover{
    border-color:var(--color-border-h)!important;color:var(--color-text)!important;
}

[data-baseweb="select"]>div{background:var(--color-surface)!important;border-color:var(--color-border)!important;color:var(--color-text)!important;}
[data-testid="stNumberInput"] input{
    background:var(--color-surface)!important;border:1px solid var(--color-border)!important;
    color:var(--color-text)!important;font-family:var(--font-sans)!important;
    border-radius:var(--r-sm)!important;padding:8px 12px!important;font-size:14px!important;
}
[data-testid="stNumberInput"] button{
    background:var(--color-surface)!important;border:1px solid var(--color-border)!important;
    color:var(--color-muted)!important;
}
hr{border:none!important;border-top:1px solid var(--color-border)!important;margin:0!important;}

.stats-row{
    display:flex;gap:1px;border:1px solid var(--color-border);border-radius:var(--r-md);
    overflow:hidden;margin:32px 48px;background:var(--color-border);
}
.stat-cell{flex:1;background:var(--color-surface);padding:20px 24px;display:flex;flex-direction:column;gap:4px;}
.stat-label{font-family:var(--font-mono);font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:0.12em;color:var(--color-muted);}
.stat-value{font-family:var(--font-serif);font-size:28px;letter-spacing:-0.03em;color:var(--color-text);line-height:1;}
.stat-value span{color:var(--color-accent);}

.section-header{display:flex;align-items:baseline;gap:12px;margin-bottom:24px;padding:0 48px;}
.section-label{font-family:var(--font-mono);font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.12em;color:var(--color-muted);}
.section-count{font-family:var(--font-mono);font-size:11px;color:var(--color-accent);background:var(--color-pill-bg);border:1px solid rgba(200,240,122,0.15);padding:2px 8px;border-radius:99px;}
.section-query{font-family:var(--font-serif);font-size:22px;font-style:italic;letter-spacing:-0.02em;color:var(--color-text);margin-left:auto;}

.img-footer{
    display:flex;align-items:center;gap:8px;
    padding:6px 2px 2px;border-top:1px solid var(--color-border);margin-top:4px;
}
.img-rank{font-family:var(--font-mono);font-size:11px;font-weight:500;color:var(--color-dimmed);}
.score-bar{flex:1;height:2px;background:var(--color-border);border-radius:99px;overflow:hidden;}
.score-fill{height:100%;background:var(--color-accent);border-radius:99px;}
.score-val{font-family:var(--font-mono);font-size:12px;font-weight:500;color:var(--color-accent);min-width:40px;text-align:right;}
.fname{font-family:var(--font-mono);font-size:10px;color:var(--color-dimmed);padding:2px 0 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;}

.empty-state{text-align:center;padding:80px 48px;color:var(--color-dimmed);}
.empty-icon{font-size:40px;margin-bottom:16px;opacity:0.4;}
.empty-text{font-size:15px;font-weight:300;line-height:1.6;color:var(--color-muted);}

[data-testid="stAlert"]{background:var(--color-surface)!important;border:1px solid var(--color-border)!important;border-radius:var(--r-md)!important;color:var(--color-text)!important;margin:0 48px!important;}

.chips-row{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:0 0 32px;padding:0 48px;}
.chip{
    font-family:var(--font-mono);font-size:11px;font-weight:500;text-transform:uppercase;
    letter-spacing:0.06em;color:var(--color-muted);background:var(--color-surface);
    border:1px solid var(--color-border);padding:6px 14px;border-radius:99px;
    cursor:default;user-select:none;
}

.app-footer{
    border-top:1px solid var(--color-border);padding:24px 48px;
    display:flex;align-items:center;justify-content:space-between;margin-top:48px;
}
.footer-l{font-family:var(--font-mono);font-size:11px;color:var(--color-dimmed);}
.footer-r{font-family:var(--font-mono);font-size:11px;color:var(--color-dimmed);}
.footer-r span{color:var(--color-muted);}

[data-testid="column"]{padding:5px!important;}
[data-testid="stImage"]{border-radius:var(--r-md);overflow:hidden;}
[data-testid="stImage"] img{border-radius:var(--r-md);display:block;width:100%;}
</style>
"""

st.markdown(GRIFFIN_CSS, unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("results", []),
    ("last_query", ""),
    ("search_time", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Cached backend loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_backend():
    if not EMBEDDINGS_FILE.exists() or not IMAGE_PATHS_FILE.exists():
        return None, None, None
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(IMAGE_PATHS_FILE, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
    embedder = CLIPEmbedder(model_name=CLIP_MODEL_NAME, device=DEVICE)
    return embedder, embeddings, image_paths


# ── Nav ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        <div class="nav-logo-mark">V</div>
        <span class="nav-logo-text">Visora</span>
    </div>
    <span class="nav-pill">Week 7 · AI Image Search</span>
</div>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Powered by CLIP &middot; openai/clip-vit-base-patch32</div>
    <h1 class="hero-title">Search images with <em>natural language</em></h1>
    <p class="hero-sub">
        Describe what you&rsquo;re looking for in plain English.
        The AI retrieves the most semantically relevant images from your dataset instantly.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Load backend (early) ───────────────────────────────────────────────────────
with st.spinner(""):
    embedder, image_embeddings, image_paths = load_backend()

backend_ready = embedder is not None
n_indexed = len(image_paths) if image_paths else 0


# ── Search form ────────────────────────────────────────────────────────────────
with st.form(key="search_form", clear_on_submit=False, border=False):
    st.markdown('<div style="max-width:720px;margin:0 auto;padding:0 48px;">', unsafe_allow_html=True)

    query_input = st.text_input(
        label="Query",
        value="",
        placeholder="Describe an image — e.g. 'a dog running on the beach'…",
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        submitted = st.form_submit_button("→ Search", use_container_width=True)
    with c2:
        top_k = st.number_input("Top K", min_value=1, max_value=24,
                                value=DEFAULT_TOP_K, step=1,
                                help="Number of images to retrieve")
    with c3:
        cleared = st.form_submit_button("✕ Clear", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Suggestion chips (display only) ───────────────────────────────────────────
SUGGESTIONS = [
    "a dog playing outside", "sunset over the ocean", "children laughing",
    "snowy mountain peak", "a cat sitting indoors", "city lights at night",
    "flowers in a garden", "a person riding a bike",
]
chips = "".join(f'<span class="chip">{s}</span>' for s in SUGGESTIONS)
st.markdown(f'<div class="chips-row">{chips}</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── Handle clear ───────────────────────────────────────────────────────────────
if cleared:
    st.session_state.results = []
    st.session_state.last_query = ""
    st.session_state.search_time = None
    st.rerun()


# ── Backend not ready ──────────────────────────────────────────────────────────
if not backend_ready:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚙️</div>
        <p class="empty-text">
            <strong style="color:#f0ece4">No embeddings found.</strong><br>
            Run <code style="color:#c8f07a;background:#1e1e22;padding:2px 6px;border-radius:4px;">
            python prepare_embeddings.py</code> first,
            then place your images in
            <code style="color:#c8f07a;background:#1e1e22;padding:2px 6px;border-radius:4px;">data/images/</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Execute search ─────────────────────────────────────────────────────────────
if submitted and query_input.strip():
    with st.spinner("Searching…"):
        t0 = time.perf_counter()
        try:
            text_emb = embedder.generate_text_embedding(query_input.strip())
            results = top_k_search(text_emb, image_embeddings, image_paths, top_k=int(top_k))
            elapsed = time.perf_counter() - t0
            st.session_state.results = results
            st.session_state.last_query = query_input.strip()
            st.session_state.search_time = elapsed
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.session_state.results = []


# ── Render results ─────────────────────────────────────────────────────────────
results = st.session_state.results

if results:
    query_display = st.session_state.last_query
    elapsed = st.session_state.search_time or 0.0
    top_score = results[0][1]

    # Stats
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-cell">
            <span class="stat-label">Results</span>
            <span class="stat-value"><span>{len(results)}</span></span>
        </div>
        <div class="stat-cell">
            <span class="stat-label">Indexed Images</span>
            <span class="stat-value">{n_indexed:,}</span>
        </div>
        <div class="stat-cell">
            <span class="stat-label">Best Match</span>
            <span class="stat-value"><span>{top_score:.1%}</span></span>
        </div>
        <div class="stat-cell">
            <span class="stat-label">Search Time</span>
            <span class="stat-value">{elapsed*1000:.0f}<span style="font-size:14px;letter-spacing:0"> ms</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section label
    st.markdown(f"""
    <div class="section-header">
        <span class="section-label">Retrieval results</span>
        <span class="section-count">{len(results)} images</span>
        <span class="section-query">&ldquo;{query_display}&rdquo;</span>
    </div>
    """, unsafe_allow_html=True)

    # Grid
    n_cols = min(4, len(results))
    cols = st.columns(n_cols, gap="small")

    for i, (img_path, score) in enumerate(results):
        col = cols[i % n_cols]
        p = Path(img_path)
        bar_pct = int(score * 100)

        with col:
            try:
                img = Image.open(img_path).convert("RGB")
                st.image(img, use_container_width=True)
            except Exception:
                st.markdown("""
                <div style="aspect-ratio:4/3;background:var(--color-surface);
                border:1px dashed var(--color-border);border-radius:12px;
                display:flex;align-items:center;justify-content:center;
                color:var(--color-dimmed);font-family:var(--font-mono);font-size:11px;">
                image not found</div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="img-footer">
                <span class="img-rank">#{i+1:02d}</span>
                <div class="score-bar" style="flex:1;margin:0 6px;">
                    <div class="score-fill" style="width:{bar_pct}%;"></div>
                </div>
                <span class="score-val">{score:.1%}</span>
            </div>
            <div class="fname" title="{p.name}">{p.name}</div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

else:
    # Welcome / empty state
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <p class="empty-text">
            Enter a text query above to search across
            <strong style="color:#f0ece4">{n_indexed:,} indexed images</strong>.<br>
            Try descriptions like <em>&ldquo;a dog on the beach&rdquo;</em>,
            <em>&ldquo;foggy forest at dawn&rdquo;</em>, or <em>&ldquo;people celebrating&rdquo;</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
    <span class="footer-l">Visora &middot; AI Image Search Engine &middot; Week 7</span>
    <span class="footer-r">
        Model <span>{CLIP_MODEL_NAME}</span> &nbsp;&middot;&nbsp;
        Device <span>{DEVICE.upper()}</span>
    </span>
</div>
""", unsafe_allow_html=True)
