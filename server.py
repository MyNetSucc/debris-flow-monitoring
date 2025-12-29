"""
Debris Flow Monitoring API Server
Serves the web interface and camera status data for Render.com deployment
Includes upload endpoints for syncing from local detection
"""
import os
import json
import secrets
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
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

# API Key for upload authentication (set via environment variable)
API_KEY = os.environ.get("SYNC_API_KEY", "")

# Ensure directories exist
SAVED_IMAGES_DIR.mkdir(exist_ok=True)

def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key for protected endpoints"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

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
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}

@app.get("/api/cctv")
@app.get("/cctv.json")
async def get_cctv():
    """Get CCTV camera list"""
    # Try full camera list first, then regular, then demo
    for filename in ["cctv_full.json", "cctv.json", "cctv_demo.json"]:
        filepath = BASE_DIR / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return JSONResponse(content=json.load(f))
    return JSONResponse(content=[])

@app.get("/camera_status.json")
async def get_status_direct():
    """Direct path for camera_status.json (for frontend compatibility)"""
    return await get_status()

# ===================== UPLOAD ENDPOINTS =====================

@app.post("/api/sync/status")
async def upload_status(
    x_api_key: str = Header(...),
    status_data: str = Form(...)
):
    """Upload camera status JSON from local detection"""
    verify_api_key(x_api_key)
    try:
        data = json.loads(status_data)
        with open(CAMERA_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"success": True, "cameras": len(data), "time": datetime.utcnow().isoformat()}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

@app.post("/api/sync/image")
async def upload_image(
    x_api_key: str = Header(...),
    camera: str = Form(...),
    filename: str = Form(...),
    image: UploadFile = File(...)
):
    """Upload detection image from local detection"""
    verify_api_key(x_api_key)
    try:
        cam_dir = SAVED_IMAGES_DIR / camera
        cam_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = cam_dir / filename
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(image.file, f)
        
        return {
            "success": True,
            "path": f"/saved_images/{camera}/{filename}",
            "time": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync/batch")
async def upload_batch(
    x_api_key: str = Header(...),
    status_data: str = Form(...),
    images: list[UploadFile] = File(default=[])
):
    """Batch upload: camera status + multiple images"""
    verify_api_key(x_api_key)
    try:
        # Save status
        data = json.loads(status_data)
        with open(CAMERA_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save images (filename format: camera_name__actual_filename.jpg)
        saved_images = []
        for img in images:
            if '__' in img.filename:
                cam, fname = img.filename.split('__', 1)
                cam_dir = SAVED_IMAGES_DIR / cam
                cam_dir.mkdir(parents=True, exist_ok=True)
                file_path = cam_dir / fname
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(img.file, f)
                saved_images.append(f"/saved_images/{cam}/{fname}")
        
        return {
            "success": True,
            "cameras": len(data),
            "images_saved": len(saved_images),
            "time": datetime.utcnow().isoformat()
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

# Mount static directories
if WEB_GIS_DIR.exists():
    app.mount("/web_gis", StaticFiles(directory=str(WEB_GIS_DIR), html=True), name="web_gis")

if SAVED_IMAGES_DIR.exists():
    app.mount("/saved_images", StaticFiles(directory=str(SAVED_IMAGES_DIR)), name="saved_images")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
