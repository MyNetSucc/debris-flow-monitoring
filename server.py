"""
Debris Flow Monitoring API Server
Production-ready version with enhanced error handling and features
"""
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
CAMERA_STATUS_FILE = BASE_DIR / "camera_status.json"
SAVED_IMAGES_DIR = BASE_DIR / "saved_images"
WEB_GIS_DIR = BASE_DIR / "web_gis"

# Stats tracking
stats = {
    "start_time": datetime.utcnow().isoformat(),
    "status_uploads": 0,
    "image_uploads": 0,
    "last_sync": None,
    "errors": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    SAVED_IMAGES_DIR.mkdir(exist_ok=True)
    logger.info(f"Server starting - Base dir: {BASE_DIR}")
    logger.info(f"Images dir: {SAVED_IMAGES_DIR}")
    yield
    # Shutdown
    logger.info("Server shutting down")

app = FastAPI(
    title="Debris Flow Monitoring API",
    description="Real-time debris flow detection and monitoring system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key
API_KEY = os.environ.get("SYNC_API_KEY", "")

def verify_api_key(x_api_key: Optional[str]) -> bool:
    """Verify API key for protected endpoints"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != API_KEY:
        stats["errors"] += 1
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def safe_read_json(filepath: Path, default=None):
    """Safely read JSON file with error handling"""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
    return default

# ===== API ROUTES =====

@app.get("/api/health")
def health():
    """Health check with detailed status"""
    camera_count = 0
    if CAMERA_STATUS_FILE.exists():
        data = safe_read_json(CAMERA_STATUS_FILE, {})
        camera_count = len(data)
    
    image_dirs = list(SAVED_IMAGES_DIR.iterdir()) if SAVED_IMAGES_DIR.exists() else []
    
    return {
        "status": "healthy",
        "time": datetime.utcnow().isoformat(),
        "api_key_set": bool(API_KEY),
        "cameras": camera_count,
        "image_folders": len([d for d in image_dirs if d.is_dir()]),
        "uptime_since": stats["start_time"]
    }

@app.get("/api/stats")
def get_stats():
    """Get server statistics"""
    return {
        **stats,
        "current_time": datetime.utcnow().isoformat()
    }

@app.get("/api/status")
def get_status():
    """Get camera status data"""
    data = safe_read_json(CAMERA_STATUS_FILE)
    if data is not None:
        return JSONResponse(content=data)
    
    # Fallback to demo
    demo_data = safe_read_json(BASE_DIR / "camera_status_demo.json", {})
    return JSONResponse(content=demo_data)

@app.get("/api/cctv")
def get_cctv():
    """Get CCTV camera list"""
    for filename in ["cctv_full.json", "cctv.json", "cctv_demo.json"]:
        data = safe_read_json(BASE_DIR / filename)
        if data is not None:
            return JSONResponse(content=data)
    return JSONResponse(content=[])

@app.get("/camera_status.json")
def get_status_json():
    """Direct path for camera_status.json (frontend compatibility)"""
    return get_status()

@app.get("/cctv.json")
def get_cctv_json():
    """Direct path for cctv.json (frontend compatibility)"""
    return get_cctv()

# ===== SYNC ROUTES =====

@app.post("/api/sync/status")
def upload_status(status_data: str = Form(...), x_api_key: Optional[str] = Header(None)):
    """Upload camera status JSON from local detection"""
    verify_api_key(x_api_key)
    try:
        data = json.loads(status_data)
        with open(CAMERA_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        stats["status_uploads"] += 1
        stats["last_sync"] = datetime.utcnow().isoformat()
        
        logger.info(f"Status uploaded: {len(data)} cameras")
        return {"success": True, "cameras": len(data), "time": stats["last_sync"]}
    except json.JSONDecodeError:
        stats["errors"] += 1
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Status upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync/image")
def upload_image(
    camera: str = Form(...),
    filename: str = Form(...),
    image: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    """Upload detection image from local detection"""
    verify_api_key(x_api_key)
    try:
        # Sanitize camera name (remove potentially dangerous characters)
        safe_camera = "".join(c for c in camera if c.isalnum() or c in '-_()（）')
        if not safe_camera:
            raise HTTPException(status_code=400, detail="Invalid camera name")
        
        cam_dir = SAVED_IMAGES_DIR / safe_camera
        cam_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '-_.')
        if not safe_filename.endswith('.jpg'):
            safe_filename += '.jpg'
        
        file_path = cam_dir / safe_filename
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(image.file, f)
        
        stats["image_uploads"] += 1
        
        return {
            "success": True, 
            "path": f"/saved_images/{safe_camera}/{safe_filename}",
            "size": file_path.stat().st_size
        }
    except HTTPException:
        raise
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
def cleanup_old_images(
    max_age_hours: int = Form(24),
    x_api_key: Optional[str] = Header(None)
):
    """Clean up old images to save disk space"""
    verify_api_key(x_api_key)
    
    if max_age_hours < 1:
        raise HTTPException(status_code=400, detail="max_age_hours must be at least 1")
    
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = 0
    
    try:
        for cam_dir in SAVED_IMAGES_DIR.iterdir():
            if not cam_dir.is_dir():
                continue
            for img_file in cam_dir.glob("*.jpg"):
                if datetime.fromtimestamp(img_file.stat().st_mtime) < cutoff:
                    img_file.unlink()
                    deleted += 1
        
        logger.info(f"Cleanup: deleted {deleted} images older than {max_age_hours}h")
        return {"success": True, "deleted": deleted}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ROOT =====

@app.get("/")
def root():
    """Serve the main web interface"""
    index_file = WEB_GIS_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return {"message": "Debris Flow Monitoring API", "docs": "/docs"}

# ===== STATIC FILES (must be last) =====

if WEB_GIS_DIR.exists():
    app.mount("/web_gis", StaticFiles(directory=str(WEB_GIS_DIR), html=True), name="web_gis")

if SAVED_IMAGES_DIR.exists():
    app.mount("/saved_images", StaticFiles(directory=str(SAVED_IMAGES_DIR)), name="saved_images")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
