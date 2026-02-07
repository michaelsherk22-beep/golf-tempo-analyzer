# golf-tempo-analyzer 🏌️⏱️

A simple Streamlit app that helps golfers calculate **swing tempo** (Address → Top → Impact) from a swing video.  
Built as a fun project to combine golf + tech.

---

## What it does

- Upload a swing video (`.mp4`, `.mov`, etc.)
- Pick (or set) the key swing moments:
  - **Address** (setup / last still moment before takeaway)
  - **Top** (top of backswing)
  - **Impact** (club strikes the ball)
- Calculates:
  - **Backswing time** (Address → Top)
  - **Downswing time** (Top → Impact)
  - **Tempo ratio** (Backswing / Downswing), e.g. **3:1**

> Many golfers target a tempo around **3:1**, but this tool is for measurement + consistency tracking (not “perfecting” your swing overnight).

---

## How to run locally

### 1) Clone the repo
```bash
git clone https://github.com/michaelsherk22-beep/golf-tempo-analyzer.git
cd golf-tempo-analyzer
