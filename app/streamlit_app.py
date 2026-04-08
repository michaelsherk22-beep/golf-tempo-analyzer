"""Golf Tempo Analyzer with browser-safe playback, stricter AI frame detection,
angle/handedness/club context, and angle-specific coaching."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st


# ───────────────────────── Tempo logic ─────────────────────────

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
        if diff < 0.3:
            return "🟢 Excellent — near the ideal 3:1!"
        elif diff < 0.7:
            return "🟡 Good — solid tempo"
        elif diff < 1.2:
            return "🟠 Fair — focus on rhythm"
        else:
            return "🔴 Needs work — tempo is off"

    @property
    def tip(self) -> str:
        if self.downswing_s <= 0:
            return ""

        if self.ratio < 2.0:
            return "💡 Your backswing looks rushed. Try a smoother takeaway and count '1-2-3' going back."
        elif self.ratio > 4.0:
            return "💡 Your downswing may be too slow relative to the backswing. Work on a cleaner, more decisive transition."
        elif self.ratio < 2.7:
            return "💡 Slightly slow down the backswing for a smoother overall rhythm."
        elif self.ratio > 3.3:
            return "💡 Nice pace overall — the downswing could be a touch more athletic."
        else:
            return "💡 Nearly perfect. Focus on repeating this tempo consistently."


def compute_tempo(address_t: float, top_t: float, impact_t: float) -> TempoMetrics:
    backswing = top_t - address_t
    downswing = impact_t - top_t
    total = impact_t - address_t
    ratio = backswing / downswing if downswing > 0 else 0.0
    return TempoMetrics(
        backswing_s=backswing,
        downswing_s=downswing,
        total_s=total,
        ratio=ratio,
    )


# ───────────────────────── Video helpers ─────────────────────────

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
            [
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-vf", "scale='min(1280,iw)':-2",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, True
        return input_path, False
    except Exception:
        return input_path, False


# ───────────────────────── Pose + AI helpers ─────────────────────────

mp_pose = mp.solutions.pose


def _interp_nan(arr: List[Optional[float]]) -> Optional[np.ndarray]:
    vals = np.array([np.nan if v is None else v for v in arr], dtype=float)
    mask = np.isnan(vals)
    if mask.all():
        return None
    vals[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), vals[~mask])
    return vals


def _smooth(x: np.ndarray, window: int = 7) -> np.ndarray:
    if len(x) < window or window < 3:
        return x
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def analyze_motion_profile(video_path: str, fps: float, max_frames: int = 450) -> Optional[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    right_wrist_y = []
    left_wrist_y = []
    right_wrist_x = []
    left_wrist_x = []
    shoulder_mid_x = []
    shoulder_mid_y = []
    hip_mid_x = []
    hip_mid_y = []
    visibility = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                rw = lms[16]
                lw = lms[15]
                rs = lms[12]
                ls = lms[11]
                rh = lms[24]
                lh = lms[23]

                right_wrist_y.append(rw.y)
                left_wrist_y.append(lw.y)
                right_wrist_x.append(rw.x)
                left_wrist_x.append(lw.x)
                shoulder_mid_x.append((rs.x + ls.x) / 2)
                shoulder_mid_y.append((rs.y + ls.y) / 2)
                hip_mid_x.append((rh.x + lh.x) / 2)
                hip_mid_y.append((rh.y + lh.y) / 2)
                visibility.append(np.mean([rw.visibility, lw.visibility, rs.visibility, ls.visibility, rh.visibility, lh.visibility]))
            else:
                right_wrist_y.append(None)
                left_wrist_y.append(None)
                right_wrist_x.append(None)
                left_wrist_x.append(None)
                shoulder_mid_x.append(None)
                shoulder_mid_y.append(None)
                hip_mid_x.append(None)
                hip_mid_y.append(None)
                visibility.append(0.0)

            frame_idx += 1

    cap.release()

    if len(visibility) < 15:
        return None

    rw_y = _interp_nan(right_wrist_y)
    lw_y = _interp_nan(left_wrist_y)
    rw_x = _interp_nan(right_wrist_x)
    lw_x = _interp_nan(left_wrist_x)
    sh_x = _interp_nan(shoulder_mid_x)
    sh_y = _interp_nan(shoulder_mid_y)
    hip_x = _interp_nan(hip_mid_x)
    hip_y = _interp_nan(hip_mid_y)

    if any(v is None for v in [rw_y, lw_y, rw_x, lw_x, sh_x, sh_y, hip_x, hip_y]):
        return None

    rw_y = _smooth(rw_y)
    lw_y = _smooth(lw_y)
    rw_x = _smooth(rw_x)
    lw_x = _smooth(lw_x)
    sh_x = _smooth(sh_x)
    sh_y = _smooth(sh_y)
    hip_x = _smooth(hip_x)
    hip_y = _smooth(hip_y)

    rw_speed = np.sqrt(np.gradient(rw_x) ** 2 + np.gradient(rw_y) ** 2) * fps
    lw_speed = np.sqrt(np.gradient(lw_x) ** 2 + np.gradient(lw_y) ** 2) * fps
    body_speed = np.sqrt(np.gradient(sh_x) ** 2 + np.gradient(sh_y) ** 2 +
                         np.gradient(hip_x) ** 2 + np.gradient(hip_y) ** 2) * fps

    motion_energy = _smooth(0.45 * rw_speed + 0.45 * lw_speed + 0.10 * body_speed, window=11)

    return {
        "visibility": np.array(visibility),
        "rw_x": rw_x, "rw_y": rw_y,
        "lw_x": lw_x, "lw_y": lw_y,
        "sh_x": sh_x, "sh_y": sh_y,
        "hip_x": hip_x, "hip_y": hip_y,
        "rw_speed": rw_speed,
        "lw_speed": lw_speed,
        "body_speed": body_speed,
        "motion_energy": motion_energy,
        "n": len(motion_energy),
    }


def detect_swing_windows(profile: Dict, fps: float) -> List[Tuple[int, int]]:
    energy = profile["motion_energy"]
    n = profile["n"]

    if n < 20:
        return []

    thresh = max(np.percentile(energy, 70), float(np.mean(energy) + 0.35 * np.std(energy)))
    active = energy > thresh

    windows = []
    start = None
    min_len = max(int(0.35 * fps), 8)

    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                windows.append((start, i - 1))
            start = None

    if start is not None and n - start >= min_len:
        windows.append((start, n - 1))

    merged = []
    for w in windows:
        if not merged:
            merged.append(list(w))
        else:
            prev = merged[-1]
            if w[0] - prev[1] <= int(0.20 * fps):
                prev[1] = w[1]
            else:
                merged.append(list(w))

    final_windows = []
    for s, e in merged:
        s = max(0, s - int(0.20 * fps))
        e = min(n - 1, e + int(0.20 * fps))
        if e - s >= min_len:
            final_windows.append((s, e))

    return final_windows[:5]


def auto_detect_key_frames(
    video_path: str,
    fps: float,
    handedness: str,
    camera_angle: str,
) -> Tuple[Optional[Tuple[int, int, int]], str, List[Tuple[int, int]]]:
    profile = analyze_motion_profile(video_path, fps)
    if profile is None:
        return None, "Could not read enough body landmarks from the clip.", []

    avg_vis = float(np.mean(profile["visibility"]))
    if avg_vis < 0.35:
        return None, "Body visibility is too low for reliable AI detection.", []

    windows = detect_swing_windows(profile, fps)
    if not windows:
        return None, "No clear swing-like motion window was detected.", []

    is_left = handedness.lower().startswith("left")

    if is_left:
        lead_x = profile["rw_x"]
        lead_y = profile["rw_y"]
        trail_x = profile["lw_x"]
        trail_y = profile["lw_y"]
        lead_speed = profile["rw_speed"]
        trail_speed = profile["lw_speed"]
    else:
        lead_x = profile["lw_x"]
        lead_y = profile["lw_y"]
        trail_x = profile["rw_x"]
        trail_y = profile["rw_y"]
        lead_speed = profile["lw_speed"]
        trail_speed = profile["rw_speed"]

    best = None
    best_score = -1.0

    for win_start, win_end in windows:
        win_len = win_end - win_start + 1
        if win_len < max(int(0.45 * fps), 10):
            continue

        energy = profile["motion_energy"][win_start:win_end + 1]
        energy_peak = int(np.argmax(energy)) + win_start

        address_search_end = min(win_start + max(int(0.35 * win_len), 5), win_end)
        low_motion_zone = profile["motion_energy"][win_start:address_search_end + 1]
        low_thresh = np.percentile(low_motion_zone, 35)
        address_candidates = np.where(low_motion_zone <= low_thresh)[0]

        if len(address_candidates) == 0:
            address_frame = win_start
        else:
            address_frame = win_start + int(address_candidates[-1])

        top_search_start = max(address_frame + 2, win_start + int(0.12 * win_len))
        top_search_end = min(win_start + int(0.65 * win_len), win_end - 2)
        if top_search_end <= top_search_start:
            continue

        top_slice_y = lead_y[top_search_start:top_search_end + 1]
        top_frame = int(np.argmin(top_slice_y)) + top_search_start

        impact_min = top_frame + max(2, int(0.08 * fps))
        impact_max = min(top_frame + int(0.60 * fps), win_end)
        if impact_max <= impact_min:
            continue

        impact_score_series = (
            0.55 * lead_speed[impact_min:impact_max + 1]
            + 0.35 * trail_speed[impact_min:impact_max + 1]
            + 0.10 * profile["body_speed"][impact_min:impact_max + 1]
        )
        impact_frame = int(np.argmax(impact_score_series)) + impact_min

        backswing_s = (top_frame - address_frame) / fps
        downswing_s = (impact_frame - top_frame) / fps

        if not (0.20 <= backswing_s <= 2.00):
            continue
        if not (0.08 <= downswing_s <= 0.70):
            continue

        ratio = backswing_s / downswing_s if downswing_s > 0 else 0
        if not (1.0 <= ratio <= 8.0):
            continue

        score = (
            float(np.max(energy)) * 0.4
            + (1.0 - abs(ratio - 3.0) / 5.0) * 0.35
            + min(backswing_s / 1.0, 1.0) * 0.15
            + avg_vis * 0.10
        )

        if camera_angle == "Behind golfer":
            score *= 0.92

        if score > best_score:
            best_score = score
            best = (address_frame, top_frame, impact_frame)

    if best is None:
        if len(windows) > 1:
            return None, "Multiple motion windows were detected, but none passed timing sanity checks. Try manual frames.", windows
        return None, "AI found motion, but not a reliable single swing. Try manual frame selection.", windows

    if len(windows) > 1:
        return best, "Multiple motion windows detected — AI chose the strongest candidate swing.", windows

    return best, "AI found one clear swing window.", windows


# ───────────────────────── Context-aware feedback ─────────────────────────

def build_context_feedback(
    camera_angle: str,
    handedness: str,
    club_type: str,
    metrics: TempoMetrics,
) -> List[str]:
    notes = []

    if camera_angle == "Face-on":
        notes.append("For a face-on view, prioritize setup balance, sway, pressure shift, and impact alignments.")
        if metrics.ratio < 2.7:
            notes.append("From face-on, the move may look rushed going back. Check whether the takeaway starts too abruptly.")
        elif metrics.ratio > 3.3:
            notes.append("From face-on, the transition may appear too soft. Look for a more athletic move from the top into impact.")
        else:
            notes.append("From face-on, the rhythm looks reasonably balanced. Focus on repeating the same transition pace.")
    elif camera_angle in ("Down-the-line (right side)", "Down-the-line (left side)"):
        notes.append("For a down-the-line view, prioritize takeaway path, hand depth, shaft plane, and delivery direction.")
        if metrics.ratio < 2.7:
            notes.append("From down-the-line, a rushed backswing can make the club get steep or disconnected early.")
        elif metrics.ratio > 3.3:
            notes.append("From down-the-line, a slow transition can leave the motion passive. Check whether the club is lingering too long at the top.")
        else:
            notes.append("From down-the-line, the tempo looks stable enough to start judging takeaway and plane more confidently.")
    else:
        notes.append("From behind golfer is less ideal than classic face-on or down-the-line, so treat technical judgments as lower confidence.")
        notes.append("This view is still useful for broad rhythm and motion-window detection, but not for fine plane details.")

    if handedness == "Left-handed":
        notes.append("All lead/trail interpretations should be mirrored for a left-handed swing.")
    else:
        notes.append("Feedback is interpreted using standard right-handed lead/trail orientation.")

    if club_type == "Driver":
        notes.append("Driver swings often look longer and wider, so the app should be a bit more forgiving on motion window length.")
    elif club_type == "Wood":
        notes.append("Fairway wood swings usually sit between driver and iron patterns, so rhythm consistency matters more than exact visual length.")
    elif club_type == "Iron":
        notes.append("Iron swings should still look athletic, but usually more compact than driver.")
    elif club_type == "Wedge":
        notes.append("Wedge swings are often shorter and more controlled, so compactness matters more than maximum arc.")

    return notes


# ───────────────────────── UI setup ─────────────────────────

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


# ───────────────────────── Header ─────────────────────────

col_logo, col_title = st.columns([1, 6])
with col_logo:
    for p in ["logo.png", "app/logo.png", "../logo.png"]:
        if os.path.exists(p):
            st.image(p, width=56)
            break

with col_title:
    st.title("Golf Swing Tempo Analyzer 🏌️")
    st.caption("Upload a swing video · pick context · use AI or manual frames · get tempo + angle-aware feedback")


# ───────────────────────── Sidebar ─────────────────────────

with st.sidebar:
    st.header("📖 How to use")
    st.markdown("""
1. **Upload** your swing video.
2. Select **camera angle**, **handedness**, and **club type**.
3. The app prepares a browser-safe version for playback.
4. Optionally click **Detect frames with AI**.
5. Review / adjust the 3 key frames.
6. Hit **Calculate**.

---

🎯 **Target:** about **3:1** backswing-to-downswing.

📱 iPhone MOV files are converted to a web-safe MP4 when possible.
""")
    st.caption("For best results, use a single swing clip with the golfer clearly visible.")


# ───────────────────────── Context inputs ─────────────────────────

st.subheader("🎯 Swing Context")

ctx1, ctx2, ctx3 = st.columns(3)
with ctx1:
    camera_angle = st.selectbox(
        "Camera angle",
        ["Face-on", "Down-the-line (right side)", "Down-the-line (left side)", "Behind golfer"],
        index=0,
    )
with ctx2:
    handedness = st.selectbox(
        "Handedness",
        ["Right-handed", "Left-handed"],
        index=0,
    )
with ctx3:
    club_type = st.selectbox(
        "Club type",
        ["Driver", "Wood", "Iron", "Wedge"],
        index=2,
    )

st.divider()


# ───────────────────────── Upload ─────────────────────────

st.subheader("📤 Upload Your Swing Video")
uploaded = st.file_uploader(
    "MP4, MOV, M4V, AVI — iPhone videos supported — up to 200 MB",
    type=["mp4", "mov", "m4v", "avi", "mpeg4"],
    label_visibility="visible",
)

if uploaded is None:
    st.info("👆 Upload a swing video above to get started.")
    st.stop()

suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    original_path = tmp.name

with st.spinner("📱 Preparing video for playback…"):
    playback_path, converted = convert_to_h264(original_path)

if converted:
    st.success("✅ Video ready to play!")
else:
    playback_path = original_path
    st.info("Using original uploaded file for playback.")

try:
    fps, total_frames = get_video_meta(original_path)
except Exception:
    fps, total_frames = 30.0, 0

st.video(playback_path)

duration_s = total_frames / fps if total_frames > 0 else 0
m1, m2, m3 = st.columns(3)
m1.metric("Frame Rate", f"{fps:.1f} fps")
m2.metric("Total Frames", str(total_frames) if total_frames > 0 else "—")
m3.metric("Duration", f"{duration_s:.1f}s" if duration_s > 0 else "—")

st.divider()


# ───────────────────────── Seconds -> frames ─────────────────────────

with st.expander("🧮 Seconds → Frame number calculator"):
    calc_sec = st.number_input(
        "Video timestamp (seconds)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.2f",
    )
    st.metric("Frame number", int(round(calc_sec * fps)))
    st.caption(f"Formula: seconds × {fps:.0f} fps = frame")


# ───────────────────────── Defaults ─────────────────────────

default_address = max(0, int(fps * 0.3))
default_top = max(default_address + 1, int(fps * 1.2))
default_impact = max(default_top + 1, int(fps * 1.5))

if total_frames > 0:
    default_address = min(default_address, max(0, total_frames - 3))
    default_top = min(default_top, max(1, total_frames - 2))
    default_impact = min(default_impact, max(2, total_frames - 1))

if "auto_frames" not in st.session_state:
    st.session_state.auto_frames = None
if "ai_message" not in st.session_state:
    st.session_state.ai_message = ""
if "swing_windows" not in st.session_state:
    st.session_state.swing_windows = []


# ───────────────────────── AI detection ─────────────────────────

st.subheader("📍 Swing Key Frames")

c1, c2 = st.columns([2, 3])

with c1:
    if st.button("🤖 Detect frames with AI"):
        with st.spinner("Analyzing swing with stricter timing rules…"):
            detected, ai_message, windows = auto_detect_key_frames(
                playback_path,
                fps,
                handedness=handedness,
                camera_angle=camera_angle,
            )
        st.session_state.auto_frames = detected
        st.session_state.ai_message = ai_message
        st.session_state.swing_windows = windows

with c2:
    if st.session_state.ai_message:
        if st.session_state.auto_frames is not None:
            a, t, i = st.session_state.auto_frames
            st.success(f"{st.session_state.ai_message} Suggested frames: Address {a}, Top {t}, Impact {i}")
        else:
            st.warning(st.session_state.ai_message)
    else:
        st.caption("Use AI detection or enter frames manually.")

if st.session_state.swing_windows:
    human_windows = []
    for idx, (s, e) in enumerate(st.session_state.swing_windows, start=1):
        human_windows.append({
            "Swing": idx,
            "Start frame": s,
            "End frame": e,
            "Start time (s)": round(s / fps, 2),
            "End time (s)": round(e / fps, 2),
        })
    with st.expander("📦 Detected motion windows"):
        st.dataframe(pd.DataFrame(human_windows), use_container_width=True)

if st.session_state.auto_frames is not None:
    default_address, default_top, default_impact = st.session_state.auto_frames


# ───────────────────────── Explanations ─────────────────────────

with st.expander("❓ What does each frame mean?"):
    st.markdown(f"""
### 🔵 Address
The last mostly still frame before the club starts moving back.

### 🟡 Top
The frame where the backswing reaches its highest point and changes direction.

### 🔴 Impact
The frame where the club strikes the ball.

### Why context matters
- **Face-on** is best for setup, shift, and impact alignments.
- **Down-the-line** is best for plane and path.
- **Behind golfer** is lower confidence for fine technical judgments.

At **{fps:.0f} fps**, the app uses timing sanity checks so AI will reject unrealistic detections instead of forcing a bad result.
""")


# ───────────────────────── Manual frames ─────────────────────────

max_f = max(total_frames - 1, 1) if total_frames > 0 else 9999

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**🔵 Address**")
    st.caption("Setup / start of backswing")
    address_frame = st.number_input(
        "Address frame",
        min_value=0,
        max_value=max_f,
        value=int(default_address),
        step=1,
        label_visibility="collapsed",
    )

with f2:
    st.markdown("**🟡 Top**")
    st.caption("Peak of backswing")
    top_frame = st.number_input(
        "Top frame",
        min_value=0,
        max_value=max_f,
        value=int(default_top),
        step=1,
        label_visibility="collapsed",
    )

with f3:
    st.markdown("**🔴 Impact**")
    st.caption("Ball contact")
    impact_frame = st.number_input(
        "Impact frame",
        min_value=0,
        max_value=max_f,
        value=int(default_impact),
        step=1,
        label_visibility="collapsed",
    )

if top_frame > address_frame and impact_frame > top_frame:
    bs_preview = (top_frame - address_frame) / fps
    ds_preview = (impact_frame - top_frame) / fps
    ratio_preview = bs_preview / ds_preview if ds_preview > 0 else 0
    st.caption(f"Preview → Backswing: {bs_preview:.2f}s · Downswing: {ds_preview:.2f}s · Ratio: {ratio_preview:.2f}:1")

st.divider()


# ───────────────────────── Calculate ─────────────────────────

if st.button("⚡ Calculate My Tempo", type="primary", use_container_width=True):
    errors = []

    if top_frame <= address_frame:
        errors.append("🔵 → 🟡 Top frame must come after Address frame.")
    if impact_frame <= top_frame:
        errors.append("🟡 → 🔴 Impact frame must come after Top frame.")

    backswing_s = (top_frame - address_frame) / fps
    downswing_s = (impact_frame - top_frame) / fps

    if not (0.08 <= downswing_s <= 1.0):
        errors.append("⏱ Downswing timing looks unrealistic. Re-check the Top and Impact frames.")
    if not (0.15 <= backswing_s <= 3.0):
        errors.append("⏱ Backswing timing looks unrealistic. Re-check the Address and Top frames.")

    for e in errors:
        st.error(e)

    if errors:
        st.stop()

    metrics = compute_tempo(address_frame / fps, top_frame / fps, impact_frame / fps)

    st.subheader("📊 Your Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Backswing", f"{metrics.backswing_s:.3f}s", help="Address → Top")
    r2.metric("Downswing", f"{metrics.downswing_s:.3f}s", help="Top → Impact")
    r3.metric("Total Swing", f"{metrics.total_s:.3f}s", help="Address → Impact")
    r4.metric("Tempo Ratio", f"{metrics.ratio:.2f}:1", help="Target is about 3:1")

    st.subheader(metrics.rating)
    st.info(metrics.tip)

    pct = min(max(metrics.ratio, 0), 6) / 6 * 100
    bar_color = "#4CAF82" if abs(metrics.ratio - 3.0) < 0.5 else "#FFD700" if abs(metrics.ratio - 3.0) < 1.0 else "#FF6B6B"
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
    chart_df = pd.DataFrame({
        "Phase": ["🔵→🟡 Backswing", "🟡→🔴 Downswing"],
        "Duration (s)": [round(metrics.backswing_s, 4), round(metrics.downswing_s, 4)],
    }).set_index("Phase")
    st.bar_chart(chart_df, color="#4CAF82")

    st.subheader("🧠 Context-aware coaching")
    feedback = build_context_feedback(camera_angle, handedness, club_type, metrics)
    for note in feedback:
        st.write(f"- {note}")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Swing #": len(st.session_state.history) + 1,
        "Camera angle": camera_angle,
        "Handedness": handedness,
        "Club": club_type,
        "Backswing (s)": round(metrics.backswing_s, 3),
        "Downswing (s)": round(metrics.downswing_s, 3),
        "Total (s)": round(metrics.total_s, 3),
        "Ratio": round(metrics.ratio, 2),
        "Rating": metrics.rating.split("—")[0].strip(),
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("📋 Session History")
        st.dataframe(
            pd.DataFrame(st.session_state.history).set_index("Swing #"),
            use_container_width=True,
        )

        st.subheader("📉 Ratio Trend")
        trend_df = pd.DataFrame({
            "Swing": [h["Swing #"] for h in st.session_state.history],
            "Ratio": [h["Ratio"] for h in st.session_state.history],
        }).set_index("Swing")
        st.line_chart(trend_df, color="#4CAF82")

    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download History CSV",
        data=csv,
        file_name="golf_tempo_history.csv",
        mime="text/csv",
    )


# ───────────────────────── Cleanup ─────────────────────────

for p in [original_path, playback_path]:
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass
