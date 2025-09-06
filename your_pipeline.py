# your_pipeline.py
"""
YouTube URL -> Transcript (official or Whisper) -> LLM summaries -> Markdown notes.

Hugging Face Spaces setup:
- packages.txt:
    ffmpeg
- requirements.txt:
    gradio>=4.0.0
    transformers
    torch
    sentencepiece
    openai-whisper
    yt-dlp>=2024.8.6
    youtube-transcript-api
"""

import re
import glob
import time
import tempfile
from pathlib import Path
from datetime import timedelta

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from transformers import pipeline
import whisper


# ======================
# Small utilities
# ======================

def _extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from many URL shapes."""
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # last resort: return any 11-char ID-looking token
    ids = re.findall(r"[A-Za-z0-9_-]{11}", url)
    if ids:
        return ids[-1]
    raise ValueError("Could not parse YouTube video ID from URL.")

def _seconds_to_hms(secs: float) -> str:
    secs = 0 if secs is None else int(secs)
    return str(timedelta(seconds=secs))


# ======================
# Transcript sources
# ======================

def _official_transcript(url: str):
    """
    Try YouTube's official captions first (English). Returns list[dict] or None.
    Adds small retry loop to ride out transient DNS/network hiccups.
    """
    try:
        vid = _extract_video_id(url)
    except Exception:
        return None

    for attempt in range(1, 4):
        try:
            data = YouTubeTranscriptApi.get_transcript(
                vid, languages=["en", "en-US", "en-GB"]
            )
            return [
                {
                    "text": x["text"].strip(),
                    "start": x["start"],
                    "duration": x["duration"],
                }
                for x in data
            ]
        except (TranscriptsDisabled, NoTranscriptFound, KeyError, ValueError):
            return None
        except Exception:
            time.sleep(1.5 * attempt)
            continue
    return None


# ======================
# Robust audio download (yt-dlp Python API, IPv4 + retries)
# ======================

def _download_audio_file(url: str, tmpdir: str) -> str:
    """
    Download best available audio using yt-dlp's Python API.
    - Forces IPv4 (avoids some IPv6/DNS quirks on hosted runners)
    - Retries on transient network errors
    Returns full path to the downloaded file (webm/m4a/mp4/etc.).
    """
    outtmpl = str(Path(tmpdir) / "audio.%(ext)s")
    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "nocheckcertificate": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "force_ipv4": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
        "geo_bypass": True,
    }

    last_err = None
    for attempt in range(1, 6):
        try:
            with YoutubeDL(base_opts) as ydl:
                ydl.download([url])
            break
        except (DownloadError, Exception) as e:
            last_err = e
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(
            f"Unable to download audio (network/DNS). Details: {type(last_err).__name__}: {last_err}"
        )

    # Find produced file (extension varies by video)
    candidates = []
    for ext in ("webm", "m4a", "mp4", "mkv", "aac", "wav", "ogg"):
        candidates.extend(glob.glob(str(Path(tmpdir) / f"audio.{ext}")))
    if not candidates:
        candidates = glob.glob(str(Path(tmpdir) / "audio.*"))
    if not candidates:
        raise RuntimeError("yt-dlp finished but no audio file was found.")
    return candidates[0]


# ======================
# Whisper transcription (lazy-load tiny model)
# ======================

_WHISPER_MODEL = None  # cache across calls

def _ensure_whisper(model_size: str = "tiny"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL

def _whisper_transcribe_url(url: str, model_size: str = "tiny"):
    """
    Download audio + transcribe with Whisper. Returns list of segments:
    [{"text", "start", "duration"}, ...]
    """
    model = _ensure_whisper(model_size)
    with tempfile.TemporaryDirectory() as td:
        audio_path = _download_audio_file(url, td)
        result = model.transcribe(audio_path)
        tr = []
        for seg in result.get("segments", []):
            tr.append(
                {
                    "text": seg.get("text", "").strip(),
                    "start": seg.get("start", 0),
                    "duration": seg.get("end", 0) - seg.get("start", 0),
                }
            )
        return tr


# ======================
# Summarization helpers (lightweight LLM)
# ======================

# Fast, CPU-friendly summarizer; upgrade to bart-large/pegasus if you need more depth.
_SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def _chunk_transcript(tr, max_words=350):
    """Group transcript items (~sentences) into chunks by word count."""
    chunks, cur, count = [], [], 0
    for item in tr:
        words = item["text"].split()
        if count + len(words) > max_words and cur:
            chunks.append(cur)
            cur = []
            count = 0
        cur.append(item)
        count += len(words)
    if cur:
        chunks.append(cur)
    return chunks

def _summarize_chunks(tr_chunks):
    outs = []
    for ch in tr_chunks:
        text = " ".join([x["text"] for x in ch])[:6000]  # guardrail for model input
        summ = _SUMMARIZER(text, max_length=160, min_length=60, do_sample=False)
        outs.append(summ[0]["summary_text"].strip())
    return outs

def _build_outline(tr_chunks):
    outline = []
    for i, ch in enumerate(tr_chunks, 1):
        start = ch[0]["start"]
        end = ch[-1]["start"] + ch[-1]["duration"]
        outline.append((i, _seconds_to_hms(start), _seconds_to_hms(end)))
    return outline

def _pick_quotes(tr, k=5):
    lines = sorted(
        [x["text"] for x in tr if len(x["text"].split()) > 8],
        key=len,
        reverse=True,
    )
    out = []
    for ln in lines:
        if len(out) >= k:
            break
        if ln not in out:
            out.append(ln)
    return out

def _make_markdown(url, transcript, tr_chunks, chunk_summaries, outline, quotes, used_whisper: bool):
    dur = (
        _seconds_to_hms(transcript[-1]["start"] + transcript[-1]["duration"])
        if transcript
        else "N/A"
    )
    key_takes = "\n".join([f"- {s}" for s in chunk_summaries[:7]])
    sections = "\n".join([f"### Section {i+1}\n{s}\n" for i, s in enumerate(chunk_summaries)])
    outline_md = "\n".join([f"- Section {i}: {a} – {b}" for (i, a, b) in outline])
    quotes_md = "\n".join([f"> {q}" for q in quotes])
    path_note = (
        "Transcript source: Whisper" if used_whisper else "Transcript source: Official YouTube captions"
    )

    return f"""# Video Notes
Source: {url}
Approx Duration: {dur}

## Key Takeaways
{key_takes}

## Outline (timestamps)
{outline_md}

## Highlights & Quotes
{quotes_md}

## Section Summaries
{sections}

---
_{path_note}_
"""


# ======================
# Public entrypoint
# ======================

def generate_notes(url: str) -> str:
    """
    Main pipeline used by the Gradio app.
    Returns Markdown notes string (or a helpful error message).
    """
    if not url or "http" not in url:
        return "❌ Please provide a valid YouTube link."

    try:
        # 1) Try official captions
        transcript = _official_transcript(url)
        used_whisper = False

        # 2) Fallback to Whisper if needed
        if not transcript:
            used_whisper = True
            transcript = _whisper_transcribe_url(url, model_size="tiny")

        if not transcript:
            return "⚠️ Could not obtain a transcript for this video."

        # 3) Chunk + summarize
        tr_chunks = _chunk_transcript(transcript, max_words=350)
        chunk_summaries = _summarize_chunks(tr_chunks)

        # 4) Outline + quotes + Markdown
        outline = _build_outline(tr_chunks)
        quotes = _pick_quotes(transcript, k=5)
        md = _make_markdown(
            url, transcript, tr_chunks, chunk_summaries, outline, quotes, used_whisper
        )
        return md

    except Exception as e:
        return f"⚠️ Error: {type(e).__name__}: {e}"
