# golf-tempo-analyzer 🏌️⏱️

A simple Streamlit app that helps golfers calculate **swing tempo** (Address → Top → Impact) from a swing video.  
Built as a fun project to combine golf + tech.

---

## What it does

Golf Tempo Analyzer calculates swing tempo using three events:

- **Address** (start of backswing)
- **Top** (top of backswing)
- **Impact** (ball strike)

It outputs:

- Address → Top (backswing time)
- Top → Impact (downswing time)
- Address → Impact (total time)
- Tempo ratio (backswing : downswing)

This Streamlit version is **manual** (you enter the frame numbers), which makes it lightweight and easy to deploy.

---

## How to run locally

### 1) Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows PowerShell


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

S

Compare multiple swings + trend charts

Add “recommended tempo range” and consistency score
