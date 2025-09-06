# app.py
import gradio as gr
from pathlib import Path
from your_pipeline import generate_notes

TITLE = "🎥 YouTube → Notes (LLM Summarizer)"
DESC = (
    "Paste a YouTube URL. The app fetches official captions if available, "
    "otherwise transcribes with Whisper, then summarizes with an LLM and returns clean Markdown notes."
)

def run(url_in: str):
    if not url_in or "http" not in url_in:
        return "❌ Please enter a valid YouTube URL.", None

    notes_md = generate_notes(url_in)
    fn = Path("notes.md")
    try:
        fn.write_text(notes_md, encoding="utf-8")
        return notes_md, str(fn)
    except Exception as e:
        return f"⚠️ Error writing file: {e}", None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}\n{DESC}")
    with gr.Row():
        url = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX",
            lines=1
        )
    with gr.Row():
        go = gr.Button("Generate Notes", variant="primary")
    out_md = gr.Markdown(label="Notes (Markdown)")
    out_file = gr.File(label="Download notes.md")
    go.click(fn=run, inputs=[url], outputs=[out_md, out_file])

if __name__ == "__main__":
    demo.queue().launch()

