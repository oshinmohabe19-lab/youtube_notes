# app.py  (Streamlit version)
import streamlit as st
from pathlib import Path
from your_pipeline import generate_notes  # uses your existing pipeline

st.set_page_config(page_title="YouTube → Notes (LLM Summarizer)", page_icon="🎥", layout="centered")

st.title("🎥 YouTube → Notes (LLM Summarizer)")
st.write(
    "Paste a YouTube URL. The app fetches official captions if available, "
    "otherwise transcribes with Whisper, then summarizes with an LLM and returns clean Markdown notes."
)

with st.form("yt_form"):
    url_in = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX")
    submitted = st.form_submit_button("Generate Notes")

if submitted:
    if not url_in or "http" not in url_in:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Generating notes..."):
            notes_md = generate_notes(url_in)

        # Show the notes (or error)
        if notes_md.startswith(("⚠️", "❌")):
            st.error(notes_md)
        else:
            st.markdown(notes_md)

            # Offer a download
            fn = Path("notes.md")
            try:
                fn.write_text(notes_md, encoding="utf-8")
                st.download_button(
                    label="Download notes.md",
                    data=fn.read_bytes(),
                    file_name="notes.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.warning(f"Could not write download file: {e}")

st.caption("Tip: If a specific video fails to fetch, try another link or run locally.")
