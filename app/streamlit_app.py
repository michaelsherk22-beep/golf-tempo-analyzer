from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import streamlit as st

# Ensure /src is importable on Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tempo.extract import iter_frames
from tempo.pose import PoseEstimator, wrist_y_series
from tempo.events import detect_events_from_wrist_y
from tempo.metrics import compute_tempo


st.set_page_config(page_title="Golf Swing Tempo Analyzer", layout="centered")
st.title("🏌️ Golf Swing Tempo Analyzer (Auto - MediaPipe)")
st.write("Upload a swing video. This version estimates **Address → Top → Impact** and computes tempo automatically.")

uploaded = st.file_uploader("Upload a swing video (mp4/mov)", type=["mp4", "mov", "m4v", "avi"])
max_frames = st.slider("Max frames to analyze (faster = lower)", 60, 600, 240, 30)

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    st.video(uploaded)

    with st.spinner("Extracting pose keypoints..."):
        est = PoseEstimator()
        times = []
        poses = []
        try:
            for f in iter_frames(video_path, max_frames=max_frames):
                times.append(f.t)
                poses.append(est.infer(f.image_bgr))
        finally:
            est.close()

        y = wrist_y_series(poses)
        y_smooth = pd.Series(y).interpolate().bfill().ffill().to_numpy()

    with st.spinner("Detecting swing events..."):
        events = detect_events_from_wrist_y(y_smooth)
        address_t = times[events.address_idx]
        top_t = times[events.top_idx]
        impact_t = times[events.impact_idx]
        m = compute_tempo(address_t, top_t, impact_t)

    st.subheader("Results")
    st.metric("Backswing (s)", f"{m.backswing_s:.3f}")
    st.metric("Downswing (s)", f"{m.downswing_s:.3f}")
    st.metric("Tempo Ratio (backswing:downswing)", f"{m.ratio:.2f}:1")

    df = pd.DataFrame({"t": times, "wrist_y": y_smooth})
    st.line_chart(df.set_index("t"))

    st.write("Detected frames:")
    st.json(
        {
            "address": {"idx": events.address_idx, "t": address_t},
            "top": {"idx": events.top_idx, "t": top_t},
            "impact": {"idx": events.impact_idx, "t": impact_t},
        }
    )

    try:
        os.remove(video_path)
    except OSError:
        pass
