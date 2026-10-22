import streamlit as st
import os

st.set_page_config(
    page_title="Python ML Cheat Sheets",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 6px 0px;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #ff4b4b;
    }

    /* Main content */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 900px;
    }

    /* GitHub banner */
    .github-banner {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    .github-banner p {
        margin: 0;
        color: #e0e0e0;
        font-size: 15px;
    }
    .star-btn {
        background-color: #238636;
        color: white !important;
        padding: 8px 20px;
        border-radius: 6px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 14px;
        white-space: nowrap;
    }
    .star-btn:hover {
        background-color: #2ea043;
    }

    /* Code blocks */
    code {
        font-size: 13.5px !important;
    }

    /* Divider */
    hr {
        border-color: #30363d;
        margin: 1.2rem 0;
    }

    /* h2 headers */
    h2 {
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-top: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

CHEATSHEETS = {
    "🔢  NumPy":               "cheatsheets/01_numpy.md",
    "🐼  Pandas":              "cheatsheets/02_pandas.md",
    "⚙️  Scikit-learn":        "cheatsheets/03_scikit_learn.md",
    "📊  Matplotlib & Seaborn":"cheatsheets/04_matplotlib_seaborn.md",
    "🧹  Data Preprocessing":  "cheatsheets/05_data_preprocessing.md",
    "🛠️  Feature Engineering": "cheatsheets/06_feature_engineering.md",
    "📈  Model Evaluation":    "cheatsheets/07_model_evaluation.md",
    "🧠  Deep Learning (Keras)":"cheatsheets/08_deep_learning_keras.md",
    "📐  Statistics for ML":   "cheatsheets/09_statistics_for_ml.md",
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🤖 Python ML Cheat Sheets")
st.sidebar.markdown("*No fluff. Just code.*")
st.sidebar.markdown("---")

selected = st.sidebar.radio(
    "Choose a topic:",
    list(CHEATSHEETS.keys()),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "⭐ [Star on GitHub](https://github.com/Kavi274/python-ml-cheatsheets)  \n"
    "🔀 [Fork & Contribute](https://github.com/Kavi274/python-ml-cheatsheets/fork)"
)
st.sidebar.markdown("Made by [Kavi274](https://github.com/Kavi274)")

# ── GitHub Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="github-banner">
    <p>📌 &nbsp; Found this useful? A ⭐ on GitHub helps others discover it.</p>
    <a class="star-btn" href="https://github.com/Kavi274/python-ml-cheatsheets" target="_blank">
        ⭐ &nbsp; Star on GitHub
    </a>
</div>
""", unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────────────────
file_path = CHEATSHEETS[selected]

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.error(f"File not found: `{file_path}`")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color: #666; font-size: 13px;'>"
    "Python ML Cheat Sheets · MIT License · "
    "<a href='https://github.com/Kavi274/python-ml-cheatsheets' target='_blank'>GitHub</a>"
    "</center>",
    unsafe_allow_html=True
)
