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
