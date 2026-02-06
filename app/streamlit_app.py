from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class Markers:
    address: Optional[int] = None
    top: Optional[int] = None
    impact: Optional[int] = None


def get_video_meta(video_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), int(frames)


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError("Could not read frame.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def tempo_from_frames(address: int, top: int, impact: int, fps: float) -> dict:
    address_t = address / fps
    top_t = top / fps
    impact_t = impact / fps
    backswing = max(0.0, top_t - address_t)
    downswing = max(1e-9, impact_t - top_t)
    ratio = backswing / downswing
    return {
        "fps": fps,
        "address": {"frame": address, "t": address_t},
        "top": {"frame": top, "t": top_t},
        "impact": {"frame": impact, "t": impact_t},
        "backswing_s": backswing,
        "downswing_s": downswing,
        "ratio": ratio,
    }


st.set_page_config(page_title="Golf Swing Tempo Analyzer", layout="centered")
st.title("🏌️ Golf Swing Tempo Analyzer (Manual Markers)")
st.write(
    "Upload a swing video, scrub to the right frames, then click **Set Address**, **Set Top**, and **Set Impact**. "
    "This version deploys cleanly on Python 3.13."
)

if "markers" not in st.session_state:
    st.session_state.markers = Markers()

uploaded = st.file_uploader("Upload a swing video (mp4/mov)", type=["mp4", "mov", "m4v", "avi"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    st.video(uploaded)

    fps, n_frames = get_video_meta(video_path)
    if n_frames <= 0:
        st.error("Could not determine total frames for this video.")
        st.stop()

    st.caption(f"Detected: {n_frames} frames @ ~{fps:.2f} FPS")

    colA, colB = st.columns([2, 1])

    with colA:
        frame_idx = st.slider("Scrub frames", 0, max(0, n_frames - 1), 0, 1)
        frame_rgb = read_frame(video_path, frame_idx)
        st.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

    with colB:
        st.subheader("Markers")
        m: Markers = st.session_state.markers

        if st.button("Set Address"):
            m.address = frame_idx
        if st.button("Set Top"):
            m.top = frame_idx
        if st.button("Set Impact"):
            m.impact = frame_idx
        if st.button("Reset markers"):
            st.session_state.markers = Markers()
            m = st.session_state.markers

        st.write({"address": m.address, "top": m.top, "impact": m.impact})

    m = st.session_state.markers
    if m.address is not None and m.top is not None and m.impact is not None:
        if not (m.address < m.top < m.impact):
            st.error("Markers must be in order: Address < Top < Impact. Adjust and try again.")
        else:
            result = tempo_from_frames(m.address, m.top, m.impact, fps)
            st.subheader("Results")
            st.metric("Backswing (s)", f"{result['backswing_s']:.3f}")
            st.metric("Downswing (s)", f"{result['downswing_s']:.3f}")
            st.metric("Tempo Ratio (backswing:downswing)", f"{result['ratio']:.2f}:1")

            df = pd.DataFrame(
                [
                    {"event": "address", "frame": result["address"]["frame"], "t": result["address"]["t"]},
                    {"event": "top", "frame": result["top"]["frame"], "t": result["top"]["t"]},
                    {"event": "impact", "frame": result["impact"]["frame"], "t": result["impact"]["t"]},
                ]
            )
            st.dataframe(df, use_container_width=True)
            st.json(result)

    try:
        os.remove(video_path)
    except OSError:
        pass
