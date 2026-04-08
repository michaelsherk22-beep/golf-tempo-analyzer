"""Golf Swing Tempo Analyzer — Streamlit Cloud ready, iPhone MOV compatible."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


# ── Tempo math ────────────────────────────────────────────────────────────────

@dataclass
class TempoMetrics:
    backswing_s: float
    downswing_s: float
    total_s: float
    ratio: float

    @property
    def rating(self) -> str:
        if self.downswing_s <= 0:
            return "Invalid"
        diff = abs(self.ratio - 3.0)
        if diff < 0.3:   return "🟢 Excellent (near 3:1)"
        elif diff < 0.7: return "🟡 Good"
        elif diff < 1.2: return "🟠 Fair – work on rhythm"
        else:            return "🔴 Needs improvement"


def compute_tempo(address_t: float, top_t: float, impact_t: float) -> TempoMetrics:
    backswing = top_t - address_t
    downswing = impact_t - top_t
    total     = impact_t - address_t
    ratio     = backswing / downswing if downswing > 0 else 0.0
    return TempoMetrics(backswing_s=backswing, downswing_s=downswing,
                        total_s=total, ratio=ratio)


def get_video_meta(video_path: str) -> tuple[float, int]:
    import imageio.v3 as iio
    meta = iio.immeta(video_path, plugin="FFMPEG") or {}
    fps_raw = meta.get("fps")
    try:
        fps = float(fps_raw) if fps_raw is not None else 30.0
    except Exception:
        fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    n_raw = meta.get("nframes") or meta.get("n_frames") or meta.get("duration_frames")
    nframes = 0
    try:
        if n_raw is not None:
            n = float(n_raw)
            if math.isfinite(n) and n > 0:
                nframes = int(n)
    except Exception:
        nframes = 0

    if nframes <= 0:
        try:
            nframes = sum(1 for _ in iio.imiter(video_path, plugin="FFMPEG"))
        except Exception:
            nframes = 0

    return fps, nframes


def convert_to_h264(input_path: str) -> tuple[str, bool]:
    """Convert any video to H.264 MP4 for universal browser playback."""
    output_path = input_path.rsplit(".", 1)[0] + "_web.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-vcodec", "libx264",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",
                "-acodec", "aac",
                "-movflags", "+faststart",
                "-loglevel", "error",
                output_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path, True
    except Exception:
        pass
    return input_path, False


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Golf Tempo Analyzer",
    page_icon="🏌️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f7f3; }
[data-testid="stHeader"]           { background: transparent; }
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e0da;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.stButton > button[kind="primary"] {
    background: #01696f;
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: .55rem 1.4rem;
}
.stButton > button[kind="primary"]:hover { background: #0c4e54; }
h1, h2, h3 { font-family: "Georgia", serif; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 5])
with col_logo:
    for candidate in ["logo.png", "app/logo.png", "../logo.png"]:
        if os.path.exists(candidate):
            st.image(candidate, width=72)
            break
with col_title:
    st.title("Golf Swing Tempo Analyzer 🏌️")
    st.caption("Works with iPhone videos — upload and go")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("How to use")
    st.markdown("""
1. **Upload** your swing video (iPhone MOV or MP4).
2. **Watch** the video and note frame numbers for:
   - 🔵 **Address** — last still moment before takeaway
   - 🟡 **Top** — wrist at peak of backswing
   - 🔴 **Impact** — club strikes the ball
3. Enter those frames → **Calculate**.

---
**Target tempo:** Tour pros average **3:1**
(backswing 3× longer than downswing).

> Consistency in your ratio matters more than hitting exactly 3:1.
""")
    st.info("📱 iPhone MOVs are auto-converted so they play in any browser.")

# ── Upload ────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a swing video",
    type=["mp4", "mov", "m4v", "avi", "mpeg4"],
    help="iPhone MOV files are automatically converted for browser playback.",
)

if uploaded is None:
    st.markdown("""
    <div style="border:2px dashed #ccc;border-radius:12px;padding:40px;
                text-align:center;color:#888;margin-top:12px;">
        <span style="font-size:2.5rem">🎥</span><br>
        <strong>Upload a swing video to get started</strong><br>
        <small>MP4 · MOV · M4V · AVI &nbsp;·&nbsp; iPhone videos supported &nbsp;·&nbsp; up to 200 MB</small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Save + convert ────────────────────────────────────────────────────────────

suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    original_path = tmp.name

needs_conversion = suffix in (".mov", ".m4v", ".mpeg4")

if needs_conversion:
    with st.spinner("📱 Converting iPhone video for browser playback…"):
        playback_path, converted = convert_to_h264(original_path)
    if converted:
        st.success("✅ iPhone video converted — ready to play!")
    else:
        playback_path = original_path
        st.warning("⚠️ Auto-conversion failed. If the video looks blank, try converting to MP4 first at cloudconvert.com")
else:
    playback_path = original_path

try:
    fps, total_frames = get_video_meta(original_path)
except Exception:
    fps, total_frames = 30.0, 0

with open(playback_path, "rb") as vf:
    st.video(vf.read())

duration_s = total_frames / fps if total_frames > 0 else 0
ic = st.columns(3)
ic[0].metric("Frame Rate",   f"{fps:.1f} fps")
ic[1].metric("Total Frames", str(total_frames) if total_frames > 0 else "unknown")
ic[2].metric("Duration",     f"{duration_s:.1f} s" if duration_s > 0 else "unknown")

# ── Frame inputs ──────────────────────────────────────────────────────────────

st.divider()
st.subheader("📍 Enter swing key frames")

with st.expander("🧮 Frame number calculator (click to open)"):
    calc_sec = st.number_input("Enter a timestamp in seconds", min_value=0.0,
                               value=1.0, step=0.1, format="%.1f")
    st.metric("Estimated frame number", int(round(calc_sec * fps)))
    st.caption(f"Formula: seconds × {fps:.0f} fps = frame number")

max_f = max(total_frames - 1, 1) if total_frames > 0 else 9999
fc1, fc2, fc3 = st.columns(3)
with fc1:
    address_frame = st.number_input("🔵 Address frame", 0, max_f, 0, 1,
                                    help="Last still frame before takeaway")
with fc2:
    top_frame = st.number_input("🟡 Top frame", 0, max_f, min(30, max_f), 1,
                                help="Peak of backswing")
with fc3:
    impact_frame = st.number_input("🔴 Impact frame", 0, max_f, min(60, max_f), 1,
                                   help="Moment of ball contact")

st.divider()

# ── Calculate ─────────────────────────────────────────────────────────────────

if st.button("⚡ Calculate Tempo", type="primary", use_container_width=True):

    errors = []
    if top_frame <= address_frame:
        errors.append("**Top** frame must be after **Address** frame.")
    if impact_frame <= top_frame:
        errors.append("**Impact** frame must be after **Top** frame.")
    for e in errors:
        st.error(e)
    if errors:
        st.stop()

    m = compute_tempo(address_frame / fps, top_frame / fps, impact_frame / fps)

    st.subheader("📊 Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Backswing",   f"{m.backswing_s:.3f} s", help="Address → Top")
    r2.metric("Downswing",   f"{m.downswing_s:.3f} s", help="Top → Impact")
    r3.metric("Total Swing", f"{m.total_s:.3f} s",     help="Address → Impact")
    r4.metric("Tempo Ratio", f"{m.ratio:.2f}:1",        help="Backswing ÷ Downswing")

    st.markdown(f"### Rating: {m.rating}")

    pct       = min(max(m.ratio, 0), 6) / 6 * 100
    bar_color = ("#01696f" if abs(m.ratio - 3.0) < 0.5
                 else "#d19900" if abs(m.ratio - 3.0) < 1.0 else "#a12c7b")
    st.markdown(f"""
    <div style="margin:12px 0 4px;font-weight:600;font-size:.9rem;color:#555;">
        Tempo Ratio gauge &nbsp;(0 → 6:1)
    </div>
    <div style="background:#e8e6e0;border-radius:999px;height:18px;position:relative;">
        <div style="background:{bar_color};width:{pct:.1f}%;height:18px;border-radius:999px;"></div>
        <div style="position:absolute;left:{3/6*100:.1f}%;top:-4px;
                    width:2px;height:26px;background:#111;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:.75rem;
                color:#888;margin-top:4px;">
        <span>0:1 (too fast)</span><span>← 3:1 target →</span><span>6:1 (too slow)</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Swing Timeline")
    df = pd.DataFrame({
        "Phase":        ["Backswing (Address→Top)", "Downswing (Top→Impact)"],
        "Duration (s)": [round(m.backswing_s, 4),   round(m.downswing_s, 4)],
    }).set_index("Phase")
    st.bar_chart(df, color="#01696f")

    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Swing #":       len(st.session_state.history) + 1,
        "Backswing (s)": round(m.backswing_s, 3),
        "Downswing (s)": round(m.downswing_s, 3),
        "Total (s)":     round(m.total_s,     3),
        "Ratio":         round(m.ratio,        2),
        "Rating":        m.rating,
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("📋 Session History")
        st.dataframe(pd.DataFrame(st.session_state.history).set_index("Swing #"),
                     use_container_width=True)
        st.subheader("📉 Ratio Trend")
        trend = pd.DataFrame({
            "Swing": [h["Swing #"] for h in st.session_state.history],
            "Ratio": [h["Ratio"]   for h in st.session_state.history],
        }).set_index("Swing")
        st.line_chart(trend, color="#01696f")

    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download History CSV", data=csv,
                       file_name="golf_tempo_history.csv", mime="text/csv")

# ── Cleanup ───────────────────────────────────────────────────────────────────
for p in [original_path, playback_path]:
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass
