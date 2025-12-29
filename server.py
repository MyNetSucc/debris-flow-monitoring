"""
Debris Flow Monitoring API Server
Serves the web interface and camera status data for Render.com deployment
"""
import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Debris Flow Monitoring API")

# CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent
CAMERA_STATUS_FILE = BASE_DIR / "camera_status.json"
SAVED_IMAGES_DIR = BASE_DIR / "saved_images"
WEB_GIS_DIR = BASE_DIR / "web_gis"

# Ensure directories exist
SAVED_IMAGES_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Serve the main web interface"""
    index_file = WEB_GIS_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Debris Flow Monitoring API", "status": "running"}

@app.get("/api/status")
async def get_status():
    """Get camera status data"""
    # Try real data first, fallback to demo
    if CAMERA_STATUS_FILE.exists():
        with open(CAMERA_STATUS_FILE, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    demo_file = BASE_DIR / "camera_status_demo.json"
    if demo_file.exists():
        with open(demo_file, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content={})

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/cctv")
@app.get("/cctv.json")
async def get_cctv():
    """Get CCTV camera list"""
    cctv_file = BASE_DIR / "cctv.json"
    if cctv_file.exists():
        with open(cctv_file, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    demo_file = BASE_DIR / "cctv_demo.json"
    if demo_file.exists():
        with open(demo_file, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content=[])

@app.get("/camera_status.json")
async def get_status_direct():
    """Direct path for camera_status.json (for frontend compatibility)"""
    return await get_status()

# Mount static directories
if WEB_GIS_DIR.exists():
    app.mount("/web_gis", StaticFiles(directory=str(WEB_GIS_DIR), html=True), name="web_gis")

if SAVED_IMAGES_DIR.exists():
    app.mount("/saved_images", StaticFiles(directory=str(SAVED_IMAGES_DIR)), name="saved_images")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
