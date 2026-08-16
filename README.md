# 🌊 Debris Flow Monitoring System

Real-time debris flow detection and monitoring dashboard using CCTV cameras across Taiwan.

## 🔗 Live Demo

**[View Dashboard](https://debris-flow-monitor.onrender.com/web_gis/)**

## Features

- 🗺️ Interactive GIS map with camera locations
- 📹 Real-time CCTV status monitoring  
- ⚠️ Alert system (Red/Yellow/Green status)
- 📊 Detection statistics and charts
- 🌙 Modern dark theme UI

## Technology

- OpenLayers for mapping
- YOLO11 for debris flow detection
- FastAPI backend
- Vanilla JS/CSS with glassmorphism design

## Project layout

```
server.py                         # FastAPI dashboard / API
sync_to_cloud.py                  # upload local detections to Render
yolo11m_seg_detectfromcctv_V3.PY  # current live CCTV detector
debris_offline_inference.py       # offline image/video inference (GUI)
web_gis/                          # map dashboard
weights/yolo11m-seg.pt            # current YOLO11-seg weights
cctv.json / camera_status.json    # live camera list and status (local)
requirements.txt                  # dashboard deps (what Render installs)
requirements-detector.txt         # local inference deps (YOLO, Selenium, ...)
docs/                             # training logs / notes
old/                              # superseded scripts, models, leftovers
test_folder/                      # local test media (not in git)
```

### Running the detector locally

```bash
pip install -r requirements-detector.txt
python yolo11m_seg_detectfromcctv_V3.PY          # writes camera_status.json + saved_images/
python sync_to_cloud.py                          # (optional) mirror results to Render
```

**ChromeDriver is no longer bundled.** A checked-in driver goes stale every time
Chrome auto-updates, and a mismatch used to kill every worker thread at startup
silently. `create_driver()` now lets Selenium Manager (bundled with selenium
>=4.6) fetch and cache a driver matching the installed Chrome. If you do place a
driver at `chromedriver-win64/chromedriver.exe` it will be tried first, with
Selenium Manager as the fallback.

## Deploy Your Own

### Option 1: Render.com (Recommended)
1. Fork this repository
2. Connect to [Render.com](https://render.com)
3. Create new Web Service from your repo
4. Render will auto-detect `render.yaml`

### Option 2: Local Development
```bash
pip install -r requirements.txt
python server.py
# Open http://localhost:8000/web_gis/
```
