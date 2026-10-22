import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
    huggingface_hub.HfFolder = HfFolder

import gradio as gr
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHEATSHEETS = {
    "🔢 NumPy":                 "cheatsheets/01_numpy.md",
    "🐼 Pandas":                "cheatsheets/02_pandas.md",
    "⚙️ Scikit-learn":          "cheatsheets/03_scikit_learn.md",
    "📊 Matplotlib & Seaborn":  "cheatsheets/04_matplotlib_seaborn.md",
    "🧹 Data Preprocessing":    "cheatsheets/05_data_preprocessing.md",
    "🛠️ Feature Engineering":   "cheatsheets/06_feature_engineering.md",
    "📈 Model Evaluation":      "cheatsheets/07_model_evaluation.md",
    "🧠 Deep Learning (Keras)": "cheatsheets/08_deep_learning_keras.md",
    "📐 Statistics for ML":     "cheatsheets/09_statistics_for_ml.md",
}

def load_cheatsheet(topic):
    path = os.path.join(BASE_DIR, CHEATSHEETS[topic])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"⚠️ File not found: {path}"

default = list(CHEATSHEETS.keys())[0]

with gr.Blocks(title="Python ML Cheat Sheets") as demo:
    gr.Markdown(
        "# 🤖 Python ML Cheat Sheets\n"
        "**A complete, beginner-friendly ML reference — no fluff, just code.**\n\n"
        "⭐ [Star on GitHub](https://github.com/Kavi274/python-ml-cheatsheets) &nbsp;|&nbsp; "
        "🔀 [Fork & Contribute](https://github.com/Kavi274/python-ml-cheatsheets/fork)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Radio(
                choices=list(CHEATSHEETS.keys()),
                value=default,
                label="📚 Select Topic"
            )
        with gr.Column(scale=4):
            content = gr.Markdown(value=load_cheatsheet(default))

    topic.change(fn=load_cheatsheet, inputs=topic, outputs=content)

demo.launch(ssr_mode=False)
