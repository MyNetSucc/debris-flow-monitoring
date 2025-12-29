"""
Debris Flow Monitoring API Server
Minimal version for debugging
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Debris Flow Monitoring API")

# CORS
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
SAVED_IMAGES_DIR.mkdir(exist_ok=True)

# API Key
API_KEY = os.environ.get("SYNC_API_KEY", "")

# ===== API ROUTES =====

@app.get("/api/health")
def health():
    return {"status": "healthy", "time": datetime.utcnow().isoformat(), "key_set": bool(API_KEY)}

@app.get("/api/status")
def get_status():
    if CAMERA_STATUS_FILE.exists():
        with open(CAMERA_STATUS_FILE, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    demo_file = BASE_DIR / "camera_status_demo.json"
    if demo_file.exists():
        with open(demo_file, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content={})

@app.get("/api/cctv")
def get_cctv():
    for filename in ["cctv_full.json", "cctv.json", "cctv_demo.json"]:
        filepath = BASE_DIR / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return JSONResponse(content=json.load(f))
    return JSONResponse(content=[])

@app.get("/camera_status.json")
def get_status_json():
    return get_status()

@app.get("/cctv.json")
def get_cctv_json():
    return get_cctv()

# ===== SYNC ROUTES =====

@app.post("/api/sync/status")
def upload_status(status_data: str = Form(...), x_api_key: Optional[str] = Header(None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        data = json.loads(status_data)
        with open(CAMERA_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"success": True, "cameras": len(data)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

@app.post("/api/sync/image")
def upload_image(
    camera: str = Form(...),
    filename: str = Form(...),
    image: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    cam_dir = SAVED_IMAGES_DIR / camera
    cam_dir.mkdir(parents=True, exist_ok=True)
    file_path = cam_dir / filename
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(image.file, f)
    return {"success": True, "path": f"/saved_images/{camera}/{filename}"}

# ===== ROOT =====

@app.get("/")
def root():
    index_file = WEB_GIS_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Debris Flow API"}

# ===== STATIC FILES (must be last) =====

if WEB_GIS_DIR.exists():
    app.mount("/web_gis", StaticFiles(directory=str(WEB_GIS_DIR), html=True), name="web_gis")

if SAVED_IMAGES_DIR.exists():
    app.mount("/saved_images", StaticFiles(directory=str(SAVED_IMAGES_DIR)), name="saved_images")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
