"""Golf Swing Analyzer
Mandatory AI detection -> review -> scoring
Analyzer-first UX: Overall grade first, tempo as a sub-result
"""

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


mp_pose = mp.solutions.pose


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
            return "Excellent"
        elif diff < 0.7:
            return "Good"
        elif diff < 1.2:
            return "Fair"
        else:
            return "Needs work"

    @property
    def tip(self) -> str:
        if self.downswing_s <= 0:
            return ""
        if self.ratio < 2.0:
            return "Your backswing looks rushed. Smooth out the takeaway and give yourself more time to load."
        elif self.ratio > 4.0:
            return "Your downswing may be too passive relative to the backswing. Work on a sharper transition."
        elif self.ratio < 2.7:
            return "Slightly slow down the backswing for a smoother rhythm."
        elif self.ratio > 3.3:
            return "Good pace overall — try a slightly more athletic transition into strike."
        return "Rhythm looks solid. Focus on repeating it consistently."


def compute_tempo(setup_t: float, top_t: float, strike_t: float) -> TempoMetrics:
    backswing = top_t - setup_t
    downswing = strike_t - top_t
    total = strike_t - setup_t
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


# ───────────────────────── Pose helpers ─────────────────────────

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

    right_wrist_y, left_wrist_y = [], []
    right_wrist_x, left_wrist_x = [], []
    shoulder_mid_x, shoulder_mid_y = [], []
    hip_mid_x, hip_mid_y = [], []
    nose_x, nose_y = [], []
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
                nose = lms[0]
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
                nose_x.append(nose.x)
                nose_y.append(nose.y)
                visibility.append(np.mean([
                    nose.visibility, rw.visibility, lw.visibility,
                    rs.visibility, ls.visibility, rh.visibility, lh.visibility
                ]))
            else:
                right_wrist_y.append(None)
                left_wrist_y.append(None)
                right_wrist_x.append(None)
                left_wrist_x.append(None)
                shoulder_mid_x.append(None)
                shoulder_mid_y.append(None)
                hip_mid_x.append(None)
                hip_mid_y.append(None)
                nose_x.append(None)
                nose_y.append(None)
                visibility.append(0.0)

            frame_idx += 1

    cap.release()

    if len(visibility) < 15:
        return None

    arrays = {
        "rw_y": _interp_nan(right_wrist_y),
        "lw_y": _interp_nan(left_wrist_y),
        "rw_x": _interp_nan(right_wrist_x),
        "lw_x": _interp_nan(left_wrist_x),
        "sh_x": _interp_nan(shoulder_mid_x),
        "sh_y": _interp_nan(shoulder_mid_y),
        "hip_x": _interp_nan(hip_mid_x),
        "hip_y": _interp_nan(hip_mid_y),
        "nose_x": _interp_nan(nose_x),
        "nose_y": _interp_nan(nose_y),
    }

    if any(v is None for v in arrays.values()):
        return None

    for k in arrays:
        arrays[k] = _smooth(arrays[k])

    rw_speed = np.sqrt(np.gradient(arrays["rw_x"]) ** 2 + np.gradient(arrays["rw_y"]) ** 2) * fps
    lw_speed = np.sqrt(np.gradient(arrays["lw_x"]) ** 2 + np.gradient(arrays["lw_y"]) ** 2) * fps
    body_speed = np.sqrt(
        np.gradient(arrays["sh_x"]) ** 2 + np.gradient(arrays["sh_y"]) ** 2 +
        np.gradient(arrays["hip_x"]) ** 2 + np.gradient(arrays["hip_y"]) ** 2
    ) * fps

    motion_energy = _smooth(0.45 * rw_speed + 0.45 * lw_speed + 0.10 * body_speed, window=11)

    arrays.update({
        "visibility": np.array(visibility),
        "rw_speed": rw_speed,
        "lw_speed": lw_speed,
        "body_speed": body_speed,
        "motion_energy": motion_energy,
        "n": len(motion_energy),
    })
    return arrays


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
) -> Tuple[Optional[Tuple[int, int, int]], str, List[Tuple[int, int]], Optional[Dict]]:
    profile = analyze_motion_profile(video_path, fps)
    if profile is None:
        return None, "Could not read enough body landmarks from the clip.", [], None

    avg_vis = float(np.mean(profile["visibility"]))
    if avg_vis < 0.35:
        return None, "Body visibility is too low for reliable AI detection.", [], profile

    windows = detect_swing_windows(profile, fps)
    if not windows:
        return None, "No clear swing-like motion window was detected.", [], profile

    is_left = handedness.lower().startswith("left")
    if is_left:
        lead_y = profile["rw_y"]
        lead_speed = profile["rw_speed"]
        trail_speed = profile["lw_speed"]
    else:
        lead_y = profile["lw_y"]
        lead_speed = profile["lw_speed"]
        trail_speed = profile["rw_speed"]

    best = None
    best_score = -1.0

    for win_start, win_end in windows:
        win_len = win_end - win_start + 1
        if win_len < max(int(0.45 * fps), 10):
            continue

        setup_search_end = min(win_start + max(int(0.35 * win_len), 5), win_end)
        low_motion_zone = profile["motion_energy"][win_start:setup_search_end + 1]
        low_thresh = np.percentile(low_motion_zone, 35)
        setup_candidates = np.where(low_motion_zone <= low_thresh)[0]
        setup_frame = win_start + int(setup_candidates[-1]) if len(setup_candidates) else win_start

        top_search_start = max(setup_frame + 2, win_start + int(0.12 * win_len))
        top_search_end = min(win_start + int(0.65 * win_len), win_end - 2)
        if top_search_end <= top_search_start:
            continue

        top_frame = int(np.argmin(lead_y[top_search_start:top_search_end + 1])) + top_search_start

        strike_min = top_frame + max(2, int(0.08 * fps))
        strike_max = min(top_frame + int(0.60 * fps), win_end)
        if strike_max <= strike_min:
            continue

        strike_score_series = (
            0.55 * lead_speed[strike_min:strike_max + 1] +
            0.35 * trail_speed[strike_min:strike_max + 1] +
            0.10 * profile["body_speed"][strike_min:strike_max + 1]
        )
        strike_frame = int(np.argmax(strike_score_series)) + strike_min

        backswing_s = (top_frame - setup_frame) / fps
        downswing_s = (strike_frame - top_frame) / fps
        if not (0.20 <= backswing_s <= 2.0):
            continue
        if not (0.08 <= downswing_s <= 0.70):
            continue

        ratio = backswing_s / downswing_s if downswing_s > 0 else 0.0
        if not (1.0 <= ratio <= 8.0):
            continue

        score = (
            float(np.max(profile["motion_energy"][win_start:win_end + 1])) * 0.40 +
            (1.0 - abs(ratio - 3.0) / 5.0) * 0.35 +
            min(backswing_s / 1.0, 1.0) * 0.15 +
            avg_vis * 0.10
        )

        if camera_angle == "Behind golfer":
            score *= 0.92

        if score > best_score:
            best_score = score
            best = (setup_frame, top_frame, strike_frame)

    if best is None:
        if len(windows) > 1:
            return None, "Multiple motion windows were detected, but none passed timing sanity checks.", windows, profile
        return None, "AI found motion, but not a reliable single swing.", windows, profile

    if len(windows) > 1:
        return best, "Multiple motion windows detected — AI chose the strongest candidate swing.", windows, profile

    return best, "AI found one clear swing window.", windows, profile


# ───────────────────────── Scoring ─────────────────────────

def clamp_score(x: float) -> float:
    return float(max(0, min(100, round(x, 1))))


def get_skill_tier(score: float) -> Tuple[str, str]:
    if score >= 90:
        return "Tour Pro", "🥇"
    elif score >= 80:
        return "Scratch Player", "🔵"
    elif score >= 70:
        return "Club Champion", "🟢"
    elif score >= 60:
        return "Weekend Warrior", "⚪"
    return "In Progress", "🟠"


def score_rhythm(metrics: TempoMetrics) -> float:
    ratio_penalty = min(abs(metrics.ratio - 3.0) * 24, 60)
    backswing_penalty = 0
    downswing_penalty = 0

    if metrics.backswing_s < 0.35:
        backswing_penalty += (0.35 - metrics.backswing_s) * 60
    elif metrics.backswing_s > 1.20:
        backswing_penalty += (metrics.backswing_s - 1.20) * 30

    if metrics.downswing_s < 0.10:
        downswing_penalty += (0.10 - metrics.downswing_s) * 120
    elif metrics.downswing_s > 0.35:
        downswing_penalty += (metrics.downswing_s - 0.35) * 90

    score = 100 - ratio_penalty - backswing_penalty - downswing_penalty
    return clamp_score(score)


def score_confidence(profile: Optional[Dict], camera_angle: str) -> float:
    if profile is None:
        return 20.0

    vis = float(np.mean(profile["visibility"])) * 100
    motion_std = float(np.std(profile["motion_energy"]))
    motion_bonus = min(motion_std * 25, 12)
    angle_penalty = 8 if camera_angle == "Behind golfer" else 0

    score = vis + motion_bonus - angle_penalty
    return clamp_score(score)


def score_stability(profile: Optional[Dict], setup_frame: int, top_frame: int) -> float:
    if profile is None or top_frame <= setup_frame:
        return 40.0

    s, t = setup_frame, top_frame
    nose_dx = np.abs(profile["nose_x"][s:t + 1] - profile["nose_x"][s]).max()
    nose_dy = np.abs(profile["nose_y"][s:t + 1] - profile["nose_y"][s]).max()
    hip_dx = np.abs(profile["hip_x"][s:t + 1] - profile["hip_x"][s]).max()

    penalty = 0
    penalty += min(nose_dx * 900, 35)
    penalty += min(nose_dy * 900, 20)
    penalty += min(hip_dx * 1000, 35)

    return clamp_score(100 - penalty)


def score_mechanics(
    profile: Optional[Dict],
    setup_frame: int,
    top_frame: int,
    strike_frame: int,
    camera_angle: str,
) -> float:
    if profile is None or not (setup_frame < top_frame < strike_frame):
        return 40.0

    s, t, i = setup_frame, top_frame, strike_frame

    shoulder_turn = abs(profile["sh_x"][t] - profile["sh_x"][s])
    hip_shift = abs(profile["hip_x"][t] - profile["hip_x"][s])
    head_shift_strike = abs(profile["nose_x"][i] - profile["nose_x"][s])
    posture_loss = abs(profile["sh_y"][i] - profile["sh_y"][s])

    if camera_angle == "Face-on":
        penalty = min(head_shift_strike * 1000, 25) + min(hip_shift * 900, 30)
        reward = min(shoulder_turn * 350, 18)
        score = 82 + reward - penalty
    elif camera_angle in ("Down-the-line (right side)", "Down-the-line (left side)"):
        penalty = min(posture_loss * 1000, 30) + min(abs(profile["hip_y"][i] - profile["hip_y"][s]) * 900, 18)
        reward = min(shoulder_turn * 420, 16)
        score = 80 + reward - penalty
    else:
        penalty = min(posture_loss * 900, 22) + min(head_shift_strike * 700, 18)
        reward = min(shoulder_turn * 300, 10)
        score = 74 + reward - penalty

    return clamp_score(score)


def compute_all_scores(
    profile: Optional[Dict],
    metrics: TempoMetrics,
    setup_frame: int,
    top_frame: int,
    strike_frame: int,
    camera_angle: str,
) -> Tuple[Dict[str, float], float, str, str]:
    stability = score_stability(profile, setup_frame, top_frame)
    mechanics = score_mechanics(profile, setup_frame, top_frame, strike_frame, camera_angle)
    tempo = score_rhythm(metrics)
    confidence = score_confidence(profile, camera_angle)

    overall = clamp_score(
        (stability * 0.30) +
        (mechanics * 0.30) +
        (tempo * 0.20) +
        (confidence * 0.20)
    )

    tier, icon = get_skill_tier(overall)

    scores = {
        "Stability": stability,
        "Swing Mechanics": mechanics,
        "Tempo": tempo,
        "Detection Confidence": confidence,
    }
    return scores, overall, tier, icon


def build_context_feedback(
    camera_angle: str,
    handedness: str,
    club_type: str,
    metrics: TempoMetrics,
    scores: Dict[str, float],
) -> List[str]:
    notes = []
    weakest = min(scores, key=scores.get)

    if camera_angle == "Face-on":
        notes.append("For face-on video, the most useful checks are head movement, lateral sway, setup stability, and impact alignments.")
    elif camera_angle in ("Down-the-line (right side)", "Down-the-line (left side)"):
        notes.append("For down-the-line video, the most useful checks are posture, takeaway direction, hand path, and delivery pattern.")
    else:
        notes.append("Behind-golfer video is lower confidence for fine technical checks, so treat mechanics as broad guidance.")

    if handedness == "Left-handed":
        notes.append("Lead-side and trail-side interpretations are mirrored for a left-handed golfer.")
    else:
        notes.append("Lead-side and trail-side interpretations use standard right-handed orientation.")

    if club_type == "Driver":
        notes.append("Driver swings often look longer and wider, so the motion window can appear more extended.")
    elif club_type == "Wood":
        notes.append("Wood swings sit between driver and iron patterns; prioritize rhythm and a clean transition.")
    elif club_type == "Iron":
        notes.append("Iron swings should still be athletic, but often look more compact than driver.")
    else:
        notes.append("Wedge swings are usually shorter and more controlled, so compactness matters more than maximum arc.")

    if weakest == "Stability":
        notes.append("Your biggest improvement area is stability. Reducing head and pelvis drift should improve consistency.")
    elif weakest == "Swing Mechanics":
        notes.append("Your biggest improvement area is swing mechanics for this camera angle. Focus on the view-specific movement pattern first.")
    elif weakest == "Tempo":
        notes.append("Your biggest improvement area is tempo. Improve your backswing-to-downswing timing before chasing smaller mechanics.")
    else:
        notes.append("Your biggest improvement area is detection confidence. Cleaner framing and body visibility should make the analyzer more trustworthy.")

    if metrics.ratio < 2.7:
        notes.append("The backswing appears a little rushed relative to the downswing.")
    elif metrics.ratio > 3.3:
        notes.append("The transition appears a little too soft relative to the backswing.")

    return notes


# ───────────────────────── App setup ─────────────────────────

st.set_page_config(
    page_title="Golf Swing Analyzer",
    page_icon="🏌️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
button, [role="button"] { min-height: 44px !important; }
input[type="number"] { min-height: 44px !important; font-size: 1rem !important; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stMetricValue"] { font-size: clamp(1.05rem, 4vw, 1.45rem) !important; }
.big-card {
    padding: 1rem 1.1rem;
    border-radius: 16px;
    background: #111827;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.small-muted {
    color: rgba(255,255,255,0.72);
    font-size: 0.92rem;
}
.grade-big {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0.15rem 0 0.25rem 0;
}
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 6])
with col_logo:
    for p in ["logo.png", "app/logo.png", "../logo.png"]:
        if os.path.exists(p):
            st.image(p, width=56)
            break
with col_title:
    st.title("Golf Swing Analyzer 🏌️")
    st.caption("AI finds the swing checkpoints first, then the app grades the swing.")

with st.sidebar:
    st.header("📖 How it works")
    st.markdown("""
1. Upload one swing video.
2. Choose angle, handedness, and club.
3. Click **Detect checkpoints with AI**.
4. Review **Setup / Top / Strike**.
5. Generate the swing grade.

The app will not grade the swing until AI has found a valid sequence first.
""")


# ───────────────────────── Context ─────────────────────────

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


# ───────────────────────── Session state ─────────────────────────

if "auto_frames" not in st.session_state:
    st.session_state.auto_frames = None
if "ai_message" not in st.session_state:
    st.session_state.ai_message = ""
if "swing_windows" not in st.session_state:
    st.session_state.swing_windows = []
if "profile" not in st.session_state:
    st.session_state.profile = None
if "ai_ok" not in st.session_state:
    st.session_state.ai_ok = False
if "history" not in st.session_state:
    st.session_state.history = []


# ───────────────────────── Step 1 ─────────────────────────

st.subheader("1️⃣ Detect swing checkpoints")
if st.button("🤖 Detect checkpoints with AI", use_container_width=True):
    with st.spinner("Analyzing swing and locating Setup / Top / Strike…"):
        detected, ai_message, windows, profile = auto_detect_key_frames(
            playback_path,
            fps,
            handedness=handedness,
            camera_angle=camera_angle,
        )
    st.session_state.auto_frames = detected
    st.session_state.ai_message = ai_message
    st.session_state.swing_windows = windows
    st.session_state.profile = profile
    st.session_state.ai_ok = detected is not None

if st.session_state.ai_message:
    if st.session_state.ai_ok and st.session_state.auto_frames is not None:
        s, t, i = st.session_state.auto_frames
        st.success(f"{st.session_state.ai_message} Setup {s}, Top {t}, Strike {i}")
    else:
        st.error(st.session_state.ai_message)

if st.session_state.swing_windows:
    with st.expander("📦 Detected motion windows"):
        win_rows = []
        for idx, (s, e) in enumerate(st.session_state.swing_windows, start=1):
            win_rows.append({
                "Swing": idx,
                "Start frame": s,
                "End frame": e,
                "Start time (s)": round(s / fps, 2),
                "End time (s)": round(e / fps, 2),
            })
        st.dataframe(pd.DataFrame(win_rows), use_container_width=True)

if not st.session_state.ai_ok:
    st.warning("AI detection must succeed before the app will generate a grade.")
    st.stop()


# ───────────────────────── Step 2 ─────────────────────────

st.subheader("2️⃣ Review checkpoints")

setup_frame, top_frame, strike_frame = st.session_state.auto_frames

with st.expander("Advanced review: adjust checkpoints if needed", expanded=False):
    max_f = max(total_frames - 1, 1) if total_frames > 0 else 9999

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔵 Setup**")
        st.caption("Last still frame before takeaway")
        setup_frame = st.number_input(
            "Setup frame",
            min_value=0,
            max_value=max_f,
            value=int(setup_frame),
            step=1,
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("**🟡 Top**")
        st.caption("Top of backswing")
        top_frame = st.number_input(
            "Top frame",
            min_value=0,
            max_value=max_f,
            value=int(top_frame),
            step=1,
            label_visibility="collapsed",
        )
    with c3:
        st.markdown("**🔴 Strike**")
        st.caption("Ball strike")
        strike_frame = st.number_input(
            "Strike frame",
            min_value=0,
            max_value=max_f,
            value=int(strike_frame),
            step=1,
            label_visibility="collapsed",
        )

if top_frame > setup_frame and strike_frame > top_frame:
    bs_preview = (top_frame - setup_frame) / fps
    ds_preview = (strike_frame - top_frame) / fps
    ratio_preview = bs_preview / ds_preview if ds_preview > 0 else 0
    st.caption(f"Preview → Backswing: {bs_preview:.2f}s · Downswing: {ds_preview:.2f}s · Ratio: {ratio_preview:.2f}:1")

with st.expander("❓ What are these checkpoints?"):
    st.markdown("""
- **Setup** = the last mostly still frame before the club starts moving back.
- **Top** = the top of the backswing where direction changes.
- **Strike** = the frame where the club reaches the ball.

These checkpoints are required because the app’s grade depends on real swing event timing.
""")


# ───────────────────────── Step 3 ─────────────────────────

st.subheader("3️⃣ Generate swing grade")

if st.button("⚡ Generate Swing Grade", type="primary", use_container_width=True):
    errors = []
    if top_frame <= setup_frame:
        errors.append("Top must come after Setup.")
    if strike_frame <= top_frame:
        errors.append("Strike must come after Top.")

    backswing_s = (top_frame - setup_frame) / fps
    downswing_s = (strike_frame - top_frame) / fps

    if not (0.08 <= downswing_s <= 1.0):
        errors.append("Downswing timing looks unrealistic. Re-check Top and Strike.")
    if not (0.15 <= backswing_s <= 3.0):
        errors.append("Backswing timing looks unrealistic. Re-check Setup and Top.")

    for e in errors:
        st.error(e)

    if errors:
        st.stop()

    profile = st.session_state.profile
    if profile is None:
        profile = analyze_motion_profile(playback_path, fps)

    metrics = compute_tempo(setup_frame / fps, top_frame / fps, strike_frame / fps)
    scores, overall, tier, icon = compute_all_scores(
        profile=profile,
        metrics=metrics,
        setup_frame=setup_frame,
        top_frame=top_frame,
        strike_frame=strike_frame,
        camera_angle=camera_angle,
    )

    weakest = min(scores, key=scores.get)
    strongest = max(scores, key=scores.get)

    st.markdown(
        f"""
        <div class="big-card">
            <div class="small-muted">Overall Swing Grade</div>
            <div class="grade-big">{icon} {overall:.1f} / 100 — {tier}</div>
            <div class="small-muted">Biggest improvement area: {weakest}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stability", f"{scores['Stability']:.1f}/100")
    c2.metric("Swing Mechanics", f"{scores['Swing Mechanics']:.1f}/100")
    c3.metric("Tempo", f"{scores['Tempo']:.1f}/100")
    c4.metric("Detection Confidence", f"{scores['Detection Confidence']:.1f}/100")

    low_reasons = []
    if scores["Stability"] < 70:
        low_reasons.append(f"stability ({scores['Stability']:.1f})")
    if scores["Swing Mechanics"] < 70:
        low_reasons.append(f"swing mechanics ({scores['Swing Mechanics']:.1f})")
    if scores["Detection Confidence"] < 70:
        low_reasons.append(f"detection confidence ({scores['Detection Confidence']:.1f})")

    if low_reasons:
        st.warning(
            "Why the overall grade is lower: " + ", ".join(low_reasons) + "."
        )
    else:
        st.info(
            "Your sub-scores are fairly balanced, so the overall grade tracks closely with the rest of the analysis."
        )

    st.subheader("🎵 Tempo Result")
    tempo_icon = "🟢" if metrics.rating == "Excellent" else "🟡" if metrics.rating == "Good" else "🟠" if metrics.rating == "Fair" else "🔴"
    st.success(f"{tempo_icon} {metrics.rating} tempo")
    st.write(metrics.tip)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Backswing", f"{metrics.backswing_s:.3f}s")
    t2.metric("Downswing", f"{metrics.downswing_s:.3f}s")
    t3.metric("Total", f"{metrics.total_s:.3f}s")
    t4.metric("Ratio", f"{metrics.ratio:.2f}:1")

    st.subheader("📉 Score Profile")
    score_df = pd.DataFrame({
        "Category": list(scores.keys()),
        "Score": list(scores.values()),
    }).set_index("Category")
    st.bar_chart(score_df, color="#4CAF82")

    grade_color = "#4CAF82" if overall >= 80 else "#FFD700" if overall >= 70 else "#FF6B6B"
    st.markdown(f"""
<div style="margin:4px 0 6px;font-size:.85rem;opacity:.8;">Overall swing grade</div>
<div style="background:#1f2430;border-radius:999px;height:22px;position:relative;border:1px solid #39414f;">
  <div style="background:{grade_color};width:{overall:.1f}%;height:22px;border-radius:999px;"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:.75rem;opacity:.6;margin-top:5px;">
  <span>0</span><span>50</span><span>100</span>
</div>
""", unsafe_allow_html=True)

    st.subheader("🧠 Coaching Feedback")
    feedback = build_context_feedback(
        camera_angle=camera_angle,
        handedness=handedness,
        club_type=club_type,
        metrics=metrics,
        scores=scores,
    )
    for note in feedback:
        st.write(f"- {note}")

    st.subheader("🎯 Focus for the next swing")
    if weakest == "Stability":
        st.warning("Focus on reducing head and pelvis drift during the backswing.")
    elif weakest == "Swing Mechanics":
        st.warning("Focus on the angle-specific movement pattern first.")
    elif weakest == "Tempo":
        st.warning("Focus on improving your backswing-to-downswing timing first.")
    else:
        st.warning("Focus on cleaner video framing and better body visibility so the analyzer can trust what it sees.")

    st.session_state.history.append({
        "Swing #": len(st.session_state.history) + 1,
        "Camera angle": camera_angle,
        "Handedness": handedness,
        "Club": club_type,
        "Setup": int(setup_frame),
        "Top": int(top_frame),
        "Strike": int(strike_frame),
        "Backswing (s)": round(metrics.backswing_s, 3),
        "Downswing (s)": round(metrics.downswing_s, 3),
        "Ratio": round(metrics.ratio, 2),
        "Stability": round(scores["Stability"], 1),
        "Swing Mechanics": round(scores["Swing Mechanics"], 1),
        "Tempo": round(scores["Tempo"], 1),
        "Detection Confidence": round(scores["Detection Confidence"], 1),
        "Overall": round(overall, 1),
        "Tier": tier,
        "Biggest Improvement Area": weakest,
        "Best Area": strongest,
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("📈 Session Trend")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df.set_index("Swing #"), use_container_width=True)

        trend_df = hist_df[["Swing #", "Overall"]].set_index("Swing #")
        st.line_chart(trend_df, color="#4CAF82")

    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Session Scores CSV",
        data=csv,
        file_name="golf_swing_scores.csv",
        mime="text/csv",
    )


# ───────────────────────── Cleanup ─────────────────────────

for p in [original_path, playback_path]:
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass
