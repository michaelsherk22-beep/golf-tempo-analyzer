"""Golf Swing Tempo Analyzer — Mobile-ready, dark theme, smart presets."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass

import pandas as pd
import streamlit as st


# ─────────────────────────── Tempo logic ─────────────────────────────────────

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


# ─────────────────────────── Page setup ──────────────────────────────────────

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


# ─────────────────────────── Header ──────────────────────────────────────────

col_logo, col_title = st.columns([1, 6])
with col_logo:
    for p in ["logo.png", "app/logo.png", "../logo.png"]:
        if os.path.exists(p):
            st.image(p, width=56)
            break
with col_title:
    st.title("Golf Swing Tempo Analyzer 🏌️")
    st.caption("Upload a swing video · enter 3 frames · get your tempo")


# ─────────────────────────── Sidebar ─────────────────────────────────────────

with st.sidebar:
    st.header("📖 How to use")
    st.markdown("""
1. **Upload** your swing video (iPhone MOV or MP4 — both work).
2. **Watch** the video and pause at each key moment.
3. **Note the frame number** using the calculator.
4. **Enter** the 3 frames and hit **Calculate**.

---

### 🎯 Target: 3:1 ratio
Tour pros take **3× as long on the backswing** as the downswing.

| Event | Frame | Time |
|---|---|---|
| Address | 10 | 0.33s |
| Top | 55 | 1.83s |
| Impact | 70 | 2.33s |
| **Ratio** | | **3.0:1** ✅ |

---
📱 iPhone MOV files are **auto-converted** when you upload.
""")


# ─────────────────────────── Upload ──────────────────────────────────────────

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


# ─────────────────────────── Frame calculator ────────────────────────────────

with st.expander("🧮 Seconds → Frame number calculator"):
    calc_sec = st.number_input("Video timestamp (seconds)", min_value=0.0,
                               value=1.0, step=0.1, format="%.2f")
    st.metric("Frame number", int(round(calc_sec * fps)))
    st.caption(f"Formula: seconds × {fps:.0f} fps = frame   |   Example: 1.5s × 30 = frame 45")

st.divider()


# ─────────────────────────── Frame inputs ────────────────────────────────────

default_address = max(0, int(fps * 0.3))
default_top     = max(default_address + 1, int(fps * 1.5))
default_impact  = max(default_top + 1, int(fps * 2.0))

if total_frames > 0:
    default_address = min(default_address, total_frames - 3)
    default_top     = min(default_top,     total_frames - 2)
    default_impact  = min(default_impact,  total_frames - 1)

st.subheader("📍 Swing Key Frames")

with st.expander("❓ What is each frame? Tap here to learn"):
    st.markdown(f"""
### 🔵 Address Frame — *"The Setup"*
**What it is:** The last frame where you are **completely still** before the club starts moving backward.

**How to find it:** Scrub forward until the club just barely starts moving back. Go back one frame — that's Address.

**Why it matters:** This is your **start clock**. A shaky or rushed Address usually causes timing problems throughout the entire swing.

---

### 🟡 Top Frame — *"The Peak"*
**What it is:** The frame where your club reaches its **highest point** at the top of the backswing — the tiny pause before the downswing fires.

**How to find it:** The moment the club stops going backward and hasn't started down yet. On 30 fps this is often just 1–2 frames wide.

**Why it matters:** Address → Top = your **backswing time**. Tour pros average **0.75–1.0 seconds** here. Too fast = rushing. Too slow = loss of power.

---

### 🔴 Impact Frame — *"The Strike"*
**What it is:** The exact frame where the **club face makes contact** with the ball.

**How to find it:** The frame just as the ball starts to compress or move. At 30 fps this is usually just 1 frame.

**Why it matters:** Top → Impact = your **downswing time**. Pros average **0.21–0.30 seconds**. Backswing ÷ Downswing = your **Tempo Ratio**.

---

### 🎯 The 3:1 Rule
Tour pros have a **3:1 tempo ratio** — backswing takes exactly 3× as long as the downswing.

At **{fps:.0f} fps**, a classic 3:1 swing looks like:

| Event | Frame | Time |
|---|---|---|
| 🔵 Address | **{default_address}** | {default_address/fps:.2f}s |
| 🟡 Top | **{default_top}** | {default_top/fps:.2f}s |
| 🔴 Impact | **{default_impact}** | {default_impact/fps:.2f}s |

Backswing: **{(default_top - default_address)/fps:.2f}s** · Downswing: **{(default_impact - default_top)/fps:.2f}s** · Ratio: **{((default_top - default_address)/(default_impact - default_top)):.1f}:1**
""")

st.caption("Frames below are pre-set to a typical 3:1 tempo. Adjust them to match your video.")

max_f = max(total_frames - 1, 1) if total_frames > 0 else 9999
fc1, fc2, fc3 = st.columns(3)

with fc1:
    st.markdown("**🔵 Address**")
    st.caption("Setup / start of backswing")
    address_frame = st.number_input("Address frame", 0, max_f,
                                    value=default_address, step=1,
                                    label_visibility="collapsed")
with fc2:
    st.markdown("**🟡 Top**")
    st.caption("Peak of backswing")
    top_frame = st.number_input("Top frame", 0, max_f,
                                value=default_top, step=1,
                                label_visibility="collapsed")
with fc3:
    st.markdown("**🔴 Impact**")
    st.caption("Ball contact")
    impact_frame = st.number_input("Impact frame", 0, max_f,
                                   value=default_impact, step=1,
                                   label_visibility="collapsed")

if top_frame > address_frame and impact_frame > top_frame:
    bs = (top_frame - address_frame) / fps
    ds = (impact_frame - top_frame) / fps
    ratio_preview = bs / ds if ds > 0 else 0
    st.caption(f"Preview → Backswing: {bs:.2f}s · Downswing: {ds:.2f}s · Ratio: {ratio_preview:.2f}:1")

st.divider()


# ─────────────────────────── Calculate ───────────────────────────────────────

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


# ─────────────────────────── Cleanup ─────────────────────────────────────────

for p in [original_path, playback_path]:
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass
