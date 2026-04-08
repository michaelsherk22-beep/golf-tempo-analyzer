"""Golf Tempo Analyzer with optional AI auto-detected key frames (MediaPipe Pose)."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st


# ───────── Tempo logic ─────────

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
        if diff < 0.3:   return "🟢 Excellent — near the ideal 3:1!"
        elif diff < 0.7: return "🟡 Good — solid tempo"
        elif diff < 1.2: return "🟠 Fair — focus on rhythm"
        else:            return "🔴 Needs work — tempo is off"

    @property
    def tip(self) -> str:
        if self.downswing_s <= 0:
            return ""
        if self.ratio < 2.0:
            return "💡 Your backswing is too rushed. Try counting '1-2-3' on the way back."
        elif self.ratio > 4.0:
            return "💡 Your downswing is too slow. Fire your hips earlier to speed up the transition."
        elif self.ratio < 2.7:
            return "💡 Slightly slow down your backswing for a smoother tempo."
        elif self.ratio > 3.3:
            return "💡 Great pace — try to make your downswing just a touch more aggressive."
        else:
            return "💡 Nearly perfect! Focus on repeating this tempo consistently."


def compute_tempo(address_t: float, top_t: float, impact_t: float) -> TempoMetrics:
    backswing = top_t - address_t
    downswing = impact_t - top_t
    total = impact_t - address_t
    ratio = backswing / downswing if downswing > 0 else 0.0
    return TempoMetrics(backswing_s=backswing, downswing_s=downswing,
                        total_s=total, ratio=ratio)


def get_video_meta(video_path: str) -> Tuple[float, int]:
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


def convert_to_h264(input_path: str) -> Tuple[str, bool]:
    output_path = input_path.rsplit(".", 1)[0] + "_web.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-vcodec", "libx264", "-profile:v", "baseline",
             "-level", "3.0", "-pix_fmt", "yuv420p",
             "-acodec", "aac", "-movflags", "+faststart",
             "-loglevel", "error", output_path],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path, True
    except Exception:
        pass
    return input_path, False


# ───────── AUTO-DETECT KEY FRAMES WITH MEDIAPIPE ─────────

mp_pose = mp.solutions.pose


def auto_detect_key_frames(video_path: str, fps: float, max_frames: int = 450) -> Optional[Tuple[int, int, int]]:
    """
    Rudimentary auto detection of (address, top, impact) using lead wrist motion.

    - Address: last frame where wrist speed is below a small threshold.
    - Top: frame where vertical velocity changes from upward to downward.
    - Impact: frame with maximum wrist speed after top.

    Returns None if detection fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    wrist_y = []   # vertical position (normalized)
    wrist_speed = []

    # Assume right-handed for now → right wrist = landmark 16 (MediaPipe Pose)
    WRIST_IDX = 16

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        prev_y = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx > max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark[WRIST_IDX]
                y = lm.y  # normalized vertical position
                wrist_y.append(y)
                if prev_y is None:
                    wrist_speed.append(0.0)
                else:
                    wrist_speed.append((y - prev_y) * fps)
                prev_y = y
            else:
                wrist_y.append(None)
                wrist_speed.append(0.0)
            frame_idx += 1

    cap.release()

    n = len(wrist_y)
    if n < 10:
        return None

    # Replace None with interpolation
    ys = np.array([np.nan if v is None else v for v in wrist_y], dtype=float)
    mask = np.isnan(ys)
    if mask.all():
        return None
    ys[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ys[~mask])

    speeds = np.array(wrist_speed, dtype=float)

    # 1) Address: last low-speed frame in first 1/3 of swing
    speed_abs = np.abs(speeds)
    cutoff = max(int(0.33 * n), 10)
    low_motion_thresh = np.percentile(speed_abs[:cutoff], 40)
    address_candidates = np.where(speed_abs[:cutoff] <= low_motion_thresh)[0]
    if len(address_candidates) == 0:
        address_frame = 0
    else:
        address_frame = int(address_candidates[-1])

    # 2) Top: vertical velocity sign change from up to down
    vy = -np.gradient(ys) * fps  # screen y increases downwards
    sign = np.sign(vy)
    sign_change = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0]
    middle_start = max(address_frame + 3, int(0.15 * n))
    middle_end = int(0.8 * n)
    mid_candidates = [i for i in sign_change if middle_start <= i <= middle_end]
    if len(mid_candidates) == 0:
        mid_slice = slice(middle_start, middle_end)
        if middle_end > middle_start:
            top_frame = int(np.argmin(ys[mid_slice]) + middle_start)
        else:
            top_frame = int(n // 2)
    else:
        top_frame = int(mid_candidates[0] + 1)

    # 3) Impact: max speed after top
    after_top_start = min(top_frame + 1, n - 1)
    if after_top_start >= n - 3:
        impact_frame = n - 1
    else:
        idx_rel = int(np.argmax(speed_abs[after_top_start:]))
        impact_frame = int(after_top_start + idx_rel)

    # Safety guard
    if not (0 <= address_frame < top_frame < impact_frame < n):
        return None

    return address_frame, top_frame, impact_frame


# ───────── Page setup ─────────

st.set_page_config(
    page_title="Golf Tempo Analyzer",
    page_icon="🏌️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
button, [role="button"] { min-height: 44px !important; }
input[type="number"] { min-height: 44px !important; font-size: 1rem !important; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stMetricValue"] { font-size: clamp(1.1rem, 5vw, 1.6rem) !important; }
</style>
""", unsafe_allow_html=True)


# ───────── Header ─────────

col_logo, col_title = st.columns([1, 6])
with col_logo:
    for p in ["logo.png", "app/logo.png", "../logo.png"]:
        if os.path.exists(p):
            st.image(p, width=56)
            break
with col_title:
    st.title("Golf Swing Tempo Analyzer 🏌️")
    st.caption("Upload a swing video · (optional) let AI detect your 3 key frames · get your tempo")


# ───────── Sidebar ─────────

with st.sidebar:
    st.header("📖 How to use")
    st.markdown("""
1. **Upload** your swing video (iPhone MOV or MP4 — both work).
2. Optionally click **Detect frames with AI**.
3. Review / tweak the 3 key frame numbers.
4. Hit **Calculate** to see your tempo.

---

🎯 **Target: 3:1 ratio** — backswing is about 3× longer than downswing.

📱 iPhone MOV files are **auto-converted** when you upload.
""")


# ───────── Upload ─────────

st.subheader("📤 Upload Your Swing Video")
uploaded = st.file_uploader(
    "MP4, MOV, M4V, AVI — iPhone videos supported — up to 200 MB",
    type=["mp4", "mov", "m4v", "avi", "mpeg4"],
    label_visibility="visible",
)

if uploaded is None:
    st.info("👆 Upload a swing video above to get started. iPhone MOV files work automatically.")
    st.stop()

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
        st.warning("⚠️ Conversion failed. If video looks blank, convert to MP4 at cloudconvert.com first.")
else:
    playback_path = original_path

try:
    fps, total_frames = get_video_meta(original_path)
except Exception:
    fps, total_frames = 30.0, 0

with open(playback_path, "rb") as vf:
    st.video(vf.read())

duration_s = total_frames / fps if total_frames > 0 else 0
c1, c2, c3 = st.columns(3)
c1.metric("Frame Rate",   f"{fps:.1f} fps")
c2.metric("Total Frames", str(total_frames) if total_frames > 0 else "—")
c3.metric("Duration",     f"{duration_s:.1f}s" if duration_s > 0 else "—")

st.divider()


# ───────── Frame calculator ─────────

with st.expander("🧮 Seconds → Frame number calculator"):
    calc_sec = st.number_input("Video timestamp (seconds)", min_value=0.0,
                               value=1.0, step=0.1, format="%.2f")
    st.metric("Frame number", int(round(calc_sec * fps)))
    st.caption(f"Formula: seconds × {fps:.0f} fps = frame   |   Example: 1.5s × 30 = frame 45")

st.divider()


# ───────── Default presets ─────────

default_address = max(0, int(fps * 0.3))
default_top     = max(default_address + 1, int(fps * 1.5))
default_impact  = max(default_top + 1, int(fps * 2.0))

if total_frames > 0:
    default_address = min(default_address, total_frames - 3)
    default_top     = min(default_top,     total_frames - 2)
    default_impact  = min(default_impact,  total_frames - 1)

if "auto_frames" not in st.session_state:
    st.session_state.auto_frames = None


# ───────── AI detection button ─────────

st.subheader("📍 Swing Key Frames")

ai_col, manual_col = st.columns([2, 3])
with ai_col:
    if st.button("🤖 Detect frames with AI"):
        with st.spinner("Analyzing swing with MediaPipe Pose (10–20 seconds)…"):
            detected = auto_detect_key_frames(playback_path, fps)
        if detected is None:
            st.warning("Could not reliably detect frames. Using recommended defaults instead.")
            st.session_state.auto_frames = None
        else:
            a, t, i = detected
            st.success(f"Detected frames · Address {a}, Top {t}, Impact {i}")
            st.session_state.auto_frames = detected

with manual_col:
    if st.session_state.auto_frames is None:
        st.caption("Using recommended defaults. You can adjust them below.")
    else:
        a, t, i = st.session_state.auto_frames
        st.caption(f"AI-suggested frames: Address {a}, Top {t}, Impact {i} (you can still tweak)")


# Use AI frames if available, otherwise defaults
if st.session_state.auto_frames is not None:
    default_address, default_top, default_impact = st.session_state.auto_frames


# ───────── Help expander ─────────

with st.expander("❓ What is each frame? Tap here to learn"):
    st.markdown(f"""
### 🔵 Address Frame — *"The Setup"*
Last frame where you are **completely still** before the club starts moving back.

### 🟡 Top Frame — *"The Peak"*
Frame where the club reaches its **highest point** and stops going back.

### 🔴 Impact Frame — *"The Strike"*
Frame where the **club face hits the ball**.

At **{fps:.0f} fps**, a classic 3:1 swing might look like:

| Event | Frame | Time |
|---|---|---|
| 🔵 Address | **{default_address}** | {default_address/fps:.2f}s |
| 🟡 Top | **{default_top}** | {default_top/fps:.2f}s |
| 🔴 Impact | **{default_impact}** | {default_impact/fps:.2f}s |
""")

max_f = max(total_frames - 1, 1) if total_frames > 0 else 9999
fc1, fc2, fc3 = st.columns(3)

with fc1:
    st.markdown("**🔵 Address**")
    st.caption("Setup / start of backswing")
    address_frame = st.number_input("Address frame", 0, max_f,
                                    value=int(default_address), step=1,
                                    label_visibility="collapsed")
with fc2:
    st.markdown("**🟡 Top**")
    st.caption("Peak of backswing")
    top_frame = st.number_input("Top frame", 0, max_f,
                                value=int(default_top), step=1,
                                label_visibility="collapsed")
with fc3:
    st.markdown("**🔴 Impact**")
    st.caption("Ball contact")
    impact_frame = st.number_input("Impact frame", 0, max_f,
                                   value=int(default_impact), step=1,
                                   label_visibility="collapsed")

if top_frame > address_frame and impact_frame > top_frame:
    bs = (top_frame - address_frame) / fps
    ds = (impact_frame - top_frame) / fps
    ratio_preview = bs / ds if ds > 0 else 0
    st.caption(f"Preview → Backswing: {bs:.2f}s · Downswing: {ds:.2f}s · Ratio: {ratio_preview:.2f}:1")

st.divider()


# ───────── Calculate ─────────

if st.button("⚡ Calculate My Tempo", type="primary", use_container_width=True):

    errors = []
    if top_frame <= address_frame:
        errors.append("🔵 → 🟡  Top frame must come AFTER Address frame.")
    if impact_frame <= top_frame:
        errors.append("🟡 → 🔴  Impact frame must come AFTER Top frame.")
    for e in errors:
        st.error(e)
    if errors:
        st.stop()

    m = compute_tempo(address_frame / fps, top_frame / fps, impact_frame / fps)

    st.subheader("📊 Your Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Backswing",   f"{m.backswing_s:.3f}s", help="Address → Top")
    r2.metric("Downswing",   f"{m.downswing_s:.3f}s", help="Top → Impact")
    r3.metric("Total Swing", f"{m.total_s:.3f}s",     help="Address → Impact")
    r4.metric("Tempo Ratio", f"{m.ratio:.2f}:1",       help="Target is 3:1")

    st.subheader(m.rating)
    st.info(m.tip)

    pct       = min(max(m.ratio, 0), 6) / 6 * 100
    bar_color = ("#4CAF82" if abs(m.ratio - 3.0) < 0.5
                 else "#FFD700" if abs(m.ratio - 3.0) < 1.0 else "#FF6B6B")
    target_pct = 3 / 6 * 100
    st.markdown(f"""
<div style="margin:4px 0 6px;font-size:.85rem;opacity:.8;">Tempo Ratio · 0:1 (fastest) → 6:1 (slowest)</div>
<div style="background:#1a2e1a;border-radius:999px;height:22px;position:relative;border:1px solid #2d4a2d;">
  <div style="background:{bar_color};width:{pct:.1f}%;height:22px;border-radius:999px;"></div>
  <div style="position:absolute;left:{target_pct:.1f}%;top:-6px;width:3px;height:34px;background:#ffffff;border-radius:2px;opacity:.5;"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:.75rem;opacity:.6;margin-top:5px;">
  <span>0:1 rushed</span>
  <span>← 3:1 ideal →</span>
  <span>6:1 too slow</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Backswing vs Downswing")
    df = pd.DataFrame({
        "Phase":        ["🔵→🟡 Backswing", "🟡→🔴 Downswing"],
        "Duration (s)": [round(m.backswing_s, 4), round(m.downswing_s, 4)],
    }).set_index("Phase")
    st.bar_chart(df, color="#4CAF82")

    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Swing #":       len(st.session_state.history) + 1,
        "Backswing (s)": round(m.backswing_s, 3),
        "Downswing (s)": round(m.downswing_s, 3),
        "Total (s)":     round(m.total_s,     3),
        "Ratio":         round(m.ratio,        2),
        "Rating":        m.rating.split("—")[0].strip(),
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("📋 Session History")
        st.dataframe(pd.DataFrame(st.session_state.history).set_index("Swing #"),
                     use_container_width=True)
        st.subheader("📉 Ratio Trend")
        st.line_chart(
            pd.DataFrame({
                "Swing": [h["Swing #"] for h in st.session_state.history],
                "Ratio": [h["Ratio"]   for h in st.session_state.history],
            }).set_index("Swing"),
            color="#4CAF82",
        )

    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download History CSV", data=csv,
                       file_name="golf_tempo_history.csv", mime="text/csv")


# ───────── Cleanup ─────────

for p in [original_path, playback_path]:
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass
