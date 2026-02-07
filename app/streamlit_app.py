from __future__ import annotations

import os
import sys

# Make src/ importable on Streamlit Cloud
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from tempo.extract import get_video_meta  # fps, total_frames


@dataclass
class Markers:
    address: Optional[int] = None
    top: Optional[int] = None
    impact: Optional[int] = None


def secs(frame_idx: int, fps: float) -> float:
    return float(frame_idx) / float(fps)


def compute_tempo(address_s: float, top_s: float, impact_s: float) -> dict:
    backswing = top_s - address_s
    downswing = impact_s - top_s
    total = impact_s - address_s
    ratio = (backswing / downswing) if downswing > 0 else np.nan
    return {
        "Address→Top (backswing) s": backswing,
        "Top→Impact (downswing) s": downswing,
        "Address→Impact (total) s": total,
        "Tempo ratio (backswing:downswing)": ratio,
    }


st.set_page_config(page_title="Golf Swing Tempo Analyzer (Manual)", layout="centered")
st.title("🏌️ Golf Swing Tempo Analyzer (Manual)")
st.write(
    "Upload a swing video, then enter the **frame numbers** for **Address**, **Top**, and **Impact**. "
    "This version is deploy-friendly (no MediaPipe/OpenCV required)."
)

uploaded = st.file_uploader("Upload a swing video", type=["mp4", "mov", "m4v", "avi"])

if "markers" not in st.session_state:
    st.session_state.markers = Markers()

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    st.video(uploaded)

    fps, total_frames = get_video_meta(video_path)
    st.caption(f"Detected FPS: **{fps:.2f}** | Total frames: **{total_frames}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.markers.address = st.number_input(
            "Address frame",
            min_value=0,
            max_value=max(total_frames - 1, 0),
            value=st.session_state.markers.address or 0,
            step=1,
        )
    with col2:
        st.session_state.markers.top = st.number_input(
            "Top frame",
            min_value=0,
            max_value=max(total_frames - 1, 0),
            value=st.session_state.markers.top or 0,
            step=1,
        )
    with col3:
        st.session_state.markers.impact = st.number_input(
            "Impact frame",
            min_value=0,
            max_value=max(total_frames - 1, 0),
            value=st.session_state.markers.impact or 0,
            step=1,
        )

    m = st.session_state.markers
    if not (m.address < m.top < m.impact):
        st.warning("Make sure frame order is **Address < Top < Impact**.")
    else:
        address_t = secs(m.address, fps)
        top_t = secs(m.top, fps)
        impact_t = secs(m.impact, fps)

        metrics = compute_tempo(address_t, top_t, impact_t)

        st.subheader("Results")
        st.dataframe(pd.DataFrame([metrics]).round(3), use_container_width=True)

        st.subheader("Event Times (seconds)")
        st.write(
            {
                "Address (s)": round(address_t, 3),
                "Top (s)": round(top_t, 3),
                "Impact (s)": round(impact_t, 3),
            }
        )

