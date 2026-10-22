import gradio as gr
import os

CHEATSHEETS = {
    "🔢  NumPy":                "cheatsheets/01_numpy.md",
    "🐼  Pandas":               "cheatsheets/02_pandas.md",
    "⚙️  Scikit-learn":         "cheatsheets/03_scikit_learn.md",
    "📊  Matplotlib & Seaborn": "cheatsheets/04_matplotlib_seaborn.md",
    "🧹  Data Preprocessing":   "cheatsheets/05_data_preprocessing.md",
    "🛠️  Feature Engineering":  "cheatsheets/06_feature_engineering.md",
    "📈  Model Evaluation":     "cheatsheets/07_model_evaluation.md",
    "🧠  Deep Learning (Keras)":"cheatsheets/08_deep_learning_keras.md",
    "📐  Statistics for ML":    "cheatsheets/09_statistics_for_ml.md",
}

def load_cheatsheet(topic):
    path = CHEATSHEETS.get(topic, "")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ File not found."

CSS = """
#header { text-align: center; padding: 10px 0 4px 0; }
#star-btn { text-align: center; margin-bottom: 8px; }
#star-btn a {
    background: #238636; color: #fff !important;
    padding: 8px 22px; border-radius: 6px;
    text-decoration: none; font-weight: 600; font-size: 14px;
}
#star-btn a:hover { background: #2ea043; }
.topic-radio label { font-size: 15px !important; }
#content { min-height: 600px; }
footer { display: none !important; }
"""

default_topic = list(CHEATSHEETS.keys())[0]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="red"), css=CSS, title="Python ML Cheat Sheets") as demo:

    gr.Markdown(
        "# 🤖 Python ML Cheat Sheets\n"
        "### A complete, beginner-friendly reference — no fluff, just code.",
        elem_id="header"
    )
    gr.Markdown(
        "⭐ **[Star on GitHub](https://github.com/Kavi274/python-ml-cheatsheets)** &nbsp;|&nbsp; "
        "🔀 [Fork & Contribute](https://github.com/Kavi274/python-ml-cheatsheets/fork)",
        elem_id="star-btn"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            topic = gr.Radio(
                choices=list(CHEATSHEETS.keys()),
                value=default_topic,
                label="📚 Select Topic",
                elem_classes="topic-radio"
            )

        with gr.Column(scale=4):
            content = gr.Markdown(
                value=load_cheatsheet(default_topic),
                elem_id="content"
            )

    topic.change(fn=load_cheatsheet, inputs=topic, outputs=content)

    gr.Markdown(
        "---\n"
        "<center>Python ML Cheat Sheets · MIT License · "
        "Made by <a href='https://github.com/Kavi274'>Kavi274</a></center>"
    )

demo.launch()
