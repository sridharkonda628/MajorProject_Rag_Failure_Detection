"""
Streamlit UI for the RAG Failure Detection System.
Provides a chat-like interface for querying the RAG pipeline
and viewing diagnostic signals.
"""

import streamlit as st
import requests
import json

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Failure Detection System",
    page_icon="🔍",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .confidence-high { color: #00e676; }
    .confidence-mid { color: #ffab00; }
    .confidence-low { color: #ff5252; }
    .success-badge {
        background: #00e676;
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .failure-badge {
        background: #ff5252;
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .warning-badge {
        background: #ffab00;
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .answer-box {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .stProgress > div > div > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def get_confidence_class(confidence: float) -> str:
    if confidence > 0.8:
        return "confidence-high"
    elif confidence > 0.4:
        return "confidence-mid"
    else:
        return "confidence-low"


def get_badge_class(failure_type: str) -> str:
    if failure_type == "Success":
        return "success-badge"
    elif "Retrieval" in failure_type:
        return "failure-badge"
    else:
        return "warning-badge"


def get_decision_emoji(decision: str) -> str:
    return {
        "finalize": "✅",
        "re-retrieve": "🔄",
        "abstain": "⛔",
    }.get(decision, "❓")


# ── Header ───────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 RAG Failure Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Lightweight Diagnostic Failure Detection & Adaptive Control for Reliable RAG</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")
    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("🟢 API Server: Online")
        else:
            st.error("🔴 API Server: Error")
    except requests.ConnectionError:
        st.error("🔴 API Server: Offline")
        st.info("Start the server with:\n```\npython main.py\n```")

    st.divider()
    st.header("📊 Signal Thresholds")
    st.markdown("""
    | Signal | Threshold |
    |--------|-----------|
    | τ_sim (sr) | 0.35 |
    | τ_nli (sn) | 0.50 |
    | τ_over (so) | 0.40 |
    """)

    st.header("🎯 Decision Thresholds")
    st.markdown("""
    | Confidence | Action |
    |------------|--------|
    | > 0.80 | ✅ Finalize |
    | 0.40 - 0.80 | 🔄 Re-retrieve |
    | ≤ 0.40 | ⛔ Abstain |
    """)

    st.divider()
    st.header("💡 Sample Queries")
    sample_queries = [
        "What is quantum computing?",
        "How does photosynthesis work?",
        "Explain machine learning types",
        "What is blockchain technology?",
        "How to cook pasta?",  # out-of-domain
    ]
    for q in sample_queries:
        if st.button(q, key=f"sample_{q}"):
            st.session_state["query_input"] = q

# ── Main Query Area ──────────────────────────────────────────────────
query = st.text_input(
    "Enter your question:",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g., What is quantum computing?",
    key="query_box",
)

col_btn, col_space = st.columns([1, 5])
with col_btn:
    submit = st.button("🚀 Ask", type="primary", use_container_width=True)

# ── Process Query ────────────────────────────────────────────────────
if submit and query:
    with st.spinner("Processing query through RAG pipeline..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"query": query},
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()

                # ── Answer Section ───────────────────────────────────
                st.markdown("---")
                st.subheader("📝 Generated Answer")

                conf_class = get_confidence_class(data["confidence"])
                badge_class = get_badge_class(data["failure_type"])

                st.markdown(
                    f'<div class="answer-box">{data["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # ── Decision Banner ──────────────────────────────────
                decision_emoji = get_decision_emoji(data["decision"])
                st.markdown(
                    f"**Decision:** {decision_emoji} `{data['decision'].upper()}`"
                    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                    f'**Failure Type:** <span class="{badge_class}">{data["failure_type"]}</span>',
                    unsafe_allow_html=True,
                )

                if data["was_reretrieval"]:
                    st.info(
                        f"🔄 Re-retrieval was triggered. "
                        f"Rewritten query: *{data['rewritten_query']}*"
                    )

                # ── Diagnostic Signals ───────────────────────────────
                st.markdown("---")
                st.subheader("📊 Diagnostic Signals")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Confidence (C)</div>'
                        f'<div class="metric-value {conf_class}">{data["confidence"]:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c2:
                    sr_class = "confidence-high" if data["sr"] >= 0.35 else "confidence-low"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Retrieval Sim (sr)</div>'
                        f'<div class="metric-value {sr_class}">{data["sr"]:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c3:
                    so_class = "confidence-high" if data["so"] >= 0.40 else "confidence-low"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Lexical Overlap (so)</div>'
                        f'<div class="metric-value {so_class}">{data["so"]:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c4:
                    sn_class = "confidence-high" if data["sn"] >= 0.50 else "confidence-low"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">NLI Support (sn)</div>'
                        f'<div class="metric-value {sn_class}">{data["sn"]:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Confidence Bar ───────────────────────────────────
                st.markdown("")
                st.markdown("**Confidence Score**")
                st.progress(min(max(data["confidence"], 0.0), 1.0))

                # ── Failure Probability ──────────────────────────────
                st.markdown(f"**P(failure):** `{data['p_failure']:.4f}`")
                st.markdown(f"**Latency:** `{data['latency_ms']:.2f} ms`")

                # ── Signal Breakdown ─────────────────────────────────
                st.markdown("---")
                st.subheader("🧮 Score Breakdown")
                st.markdown(
                    f"```\n"
                    f"Confidence = α·sr + β·so + γ·sn\n"
                    f"           = 0.3×{data['sr']:.4f} + 0.2×{data['so']:.4f} + 0.5×{data['sn']:.4f}\n"
                    f"           = {0.3*data['sr']:.4f} + {0.2*data['so']:.4f} + {0.5*data['sn']:.4f}\n"
                    f"           = {data['confidence']:.4f}\n"
                    f"P(failure) = 1 - {data['confidence']:.4f} = {data['p_failure']:.4f}\n"
                    f"```"
                )

                # ── Retrieved Documents ──────────────────────────────
                st.markdown("---")
                with st.expander("📄 Retrieved Documents", expanded=False):
                    for i, (doc, score) in enumerate(
                        zip(data["retrieved_docs"], data["retrieval_scores"])
                    ):
                        st.markdown(f"**Document {i+1}** (score: `{score:.4f}`)")
                        st.text(doc[:500] + ("..." if len(doc) > 500 else ""))
                        st.divider()

                # ── Raw JSON ─────────────────────────────────────────
                with st.expander("🔧 Raw API Response", expanded=False):
                    st.json(data)

            else:
                st.error(f"API Error: {response.status_code} — {response.text}")

        except requests.ConnectionError:
            st.error(
                "❌ Cannot connect to the API server. "
                "Make sure the FastAPI backend is running on port 8000."
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif submit and not query:
    st.warning("Please enter a query.")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.85rem;">'
    'RAG Failure Detection System — Lightweight Diagnostic Failure Detection '
    '& Adaptive Control for Reliable RAG Systems'
    '</div>',
    unsafe_allow_html=True,
)
