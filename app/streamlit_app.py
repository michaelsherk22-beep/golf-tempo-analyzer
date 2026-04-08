"""Golf Swing Tempo Analyzer – Manual Frame Entry Mode."""
from __future__ import annotations
import os, tempfile
import pandas as pd
import streamlit as st
from tempo.extract import get_video_meta
from tempo.metrics import compute_tempo

st.set_page_config(page_title="Golf Tempo Analyzer", page_icon="🏌️",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f7f3; }
div[data-testid="metric-container"] {
    background: white; border: 1px solid #e2e0da; border-radius: 10px;
    padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stButton > button[kind="primary"] {
    background: #01696f; border: none; border-radius: 8px;
    color: white; font-weight: 600; padding: 0.55rem 1.4rem;
}
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    logo_path = os.path.join(os.path.dirname(__file__), "..", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=72)
with col_title:
    st.title("Golf Swing Tempo Analyzer 🏌️")
    st.caption("Manual frame mode — lightweight, no AI required")

st.divider()

with st.sidebar:
    st.header("How to use")
    st.markdown("""
1. **Upload** your swing video.
2. **Watch** the video; note frame numbers for:
   - 🔵 **Address** — last still moment before takeaway
   - 🟡 **Top** — wrist at peak of backswing
   - 🔴 **Impact** — club strikes the ball
3. Enter those frames → **Calculate**.

---
**Target tempo:** Tour pros average **3:1**
(backswing 3× longer than downswing).
""")
    st.info("Tip: Film at 60 fps+ for better precision.")

uploaded = st.file_uploader("Upload a swing video", type=["mp4", "mov", "m4v", "avi"])

if uploaded is None:
    st.markdown("""
    <div style="border:2px dashed #ccc;border-radius:12px;padding:40px;
                text-align:center;color:#888;margin-top:12px;">
        <span style="font-size:2.5rem">🎥</span><br>
        <strong>Upload a swing video to get started</strong><br>
        <small>MP4, MOV, M4V or AVI · up to 200 MB</small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    video_path = tmp.name

try:
    fps, total_frames = get_video_meta(video_path)
except Exception:
    fps, total_frames = 30.0, 0

st.video(uploaded)

duration_s = total_frames / fps if total_frames > 0 else 0
ic = st.columns(3)
ic[0].metric("Frame Rate", f"{fps:.1f} fps")
ic[1].metric("Total Frames", str(total_frames) if total_frames > 0 else "unknown")
ic[2].metric("Duration", f"{duration_s:.1f} s" if duration_s > 0 else "unknown")

st.divider()
st.subheader("📍 Enter swing key frames")
st.caption("Scrub the video above, then enter the frame number for each key moment.")

max_f = max(total_frames - 1, 1) if total_frames > 0 else 9999
fc1, fc2, fc3 = st.columns(3)
with fc1:
    address_frame = st.number_input("🔵 Address frame", 0, max_f, 0, 1)
with fc2:
    top_frame = st.number_input("🟡 Top frame", 0, max_f, min(30, max_f), 1)
with fc3:
    impact_frame = st.number_input("🔴 Impact frame", 0, max_f, min(60, max_f), 1)

st.divider()

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
    r1.metric("Backswing", f"{m.backswing_s:.3f} s")
    r2.metric("Downswing", f"{m.downswing_s:.3f} s")
    r3.metric("Total Swing", f"{m.total_s:.3f} s")
    r4.metric("Tempo Ratio", f"{m.ratio:.2f}:1")
    st.markdown(f"### Rating: {m.rating}")

    # Gauge bar
    pct = min(max(m.ratio, 0), 6) / 6 * 100
    bar_color = "#01696f" if abs(m.ratio-3)<0.5 else "#d19900" if abs(m.ratio-3)<1 else "#a12c7b"
    st.markdown(f"""
    <div style="margin:12px 0 4px;font-weight:600;font-size:.9rem;color:#555;">Tempo Ratio gauge (0–6:1)</div>
    <div style="background:#e8e6e0;border-radius:999px;height:18px;position:relative;">
      <div style="background:{bar_color};width:{pct:.1f}%;height:18px;border-radius:999px;"></div>
      <div style="position:absolute;left:{3/6*100:.1f}%;top:-4px;width:2px;height:26px;background:#111;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:.75rem;color:#888;margin-top:4px;">
      <span>0:1 (too fast)</span><span>← 3:1 target →</span><span>6:1 (too slow)</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Swing Timeline")
    df = pd.DataFrame({"Phase": ["Backswing", "Downswing"],
                        "Duration (s)": [m.backswing_s, m.downswing_s]}).set_index("Phase")
    st.bar_chart(df, color="#01696f")

    # Session history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Swing #": len(st.session_state.history) + 1,
        "Backswing (s)": round(m.backswing_s, 3),
        "Downswing (s)": round(m.downswing_s, 3),
        "Total (s)": round(m.total_s, 3),
        "Ratio": round(m.ratio, 2),
        "Rating": m.rating,
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("📋 Session History")
        st.dataframe(pd.DataFrame(st.session_state.history).set_index("Swing #"),
                     use_container_width=True)
        st.subheader("📉 Ratio Trend")
        trend = pd.DataFrame({"Swing": [h["Swing #"] for h in st.session_state.history],
                               "Ratio": [h["Ratio"] for h in st.session_state.history]
                              }).set_index("Swing")
        st.line_chart(trend, color="#01696f")

    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download History CSV", data=csv,
                        file_name="golf_tempo_history.csv", mime="text/csv")

try:
    os.remove(video_path)
except OSError:
    pass
