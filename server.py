"""
Debris Flow Monitoring API Server
Production-ready version with enhanced error handling and features
"""
import os
import csv
import json
import time
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request, Body
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
LOGS_DIR = BASE_DIR / "logs"
METRICS_DIR = LOGS_DIR / "metrics"
EVENTS_FILE = LOGS_DIR / "events.jsonl"
ACKS_FILE = LOGS_DIR / "acks.jsonl"

# Acknowledgement reason codes. The console renders the labels; the server only
# checks that the code is one it knows, so a malformed client cannot write
# arbitrary text into what is now a record of who did what about an alert.
REASON_CODES = {
    "fp_noise", "fp_lens", "unclear",
    "verified_none", "verified_debris",
    "reported", "dispatched", "other",
}
ACK_ACTIONS = {"ack", "unack", "clear_red"}
acks_lock = threading.Lock()

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
# allow_credentials=True combined with allow_origins=["*"] is invalid per the CORS
# spec and browsers reject it. Auth here is a plain X-API-Key header, not cookies,
# so credentials are not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    """
    Safely read a JSON file.

    The detector replaces camera_status.json atomically (write temp + os.replace).
    On Windows that rename briefly collides with a reader holding the file open,
    surfacing as PermissionError, so retry a couple of times before giving up
    rather than falling back to stale demo data on a transient miss.
    """
    for attempt in range(3):
        try:
            if not filepath.exists():
                return default
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except PermissionError:
            if attempt == 2:
                logger.warning(f"{filepath} locked by writer, giving up this read")
                break
            time.sleep(0.05)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
            break
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            break
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
    for filename in ["cctv.json", "cctv_demo.json"]:
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

# ===== HISTORY ROUTES =====

def _metrics_path(camera: str) -> Path:
    """
    Resolve a camera name to its metrics CSV, refusing anything that escapes
    the metrics directory. The name arrives from the URL, so '../../secrets.env'
    is the obvious attempt; resolving and then re-checking containment catches
    both that and symlink tricks.
    """
    if not camera or len(camera) > 120:
        raise HTTPException(status_code=400, detail="Invalid camera name")
    safe = camera.replace("/", "").replace("\\", "").replace("\x00", "")
    path = (METRICS_DIR / f"{safe}.csv").resolve()
    try:
        path.relative_to(METRICS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid camera name")
    return path


@app.get("/api/metrics/{camera}")
def get_metrics(camera: str, hours: float = 24.0, max_points: int = 720):
    """
    Per-camera detection history, read from the CSV the detector already writes.

    Without this the dashboard can only chart what it observed since the page
    was opened, which resets on every refresh. All 63 files together are ~4 MB,
    so this stays cheap.
    """
    path = _metrics_path(camera)
    if not path.exists():
        return {"camera": camera, "points": [], "truncated": False}

    cutoff = datetime.now() - timedelta(hours=max(0.1, min(hours, 24 * 14)))
    rows = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ts = row.get("timestamp")
                if not ts:
                    continue
                try:
                    t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue          # one unparseable row must not kill the series
                if t < cutoff:
                    continue
                rows.append((t, row))
    except OSError as e:
        logger.error(f"metrics read failed for {camera}: {e}")
        raise HTTPException(status_code=500, detail="Cannot read metrics")

    # Downsample, but never drop a row where the alert level changed — those are
    # the only rows an operator actually looks for in a long series.
    truncated = False
    if max_points > 0 and len(rows) > max_points:
        truncated = True
        step = len(rows) / max_points
        keep, prev_alert, nxt = [], None, 0.0
        for i, (t, row) in enumerate(rows):
            alert = row.get("alert")
            if alert != prev_alert or i >= nxt or i == len(rows) - 1:
                keep.append((t, row))
                nxt = i + step
            prev_alert = alert
        rows = keep

    def num(v):
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return 0.0

    points = [{
        "t": t.strftime("%Y-%m-%d %H:%M:%S"),
        "alert": row.get("alert") or "green",
        "prop": {k: num(row.get(f"{k}_prop")) for k in
                 ("clear_water", "muddy_water", "debris_flow", "large_rock")},
        "conf": {k: num(row.get(f"{k}_conf")) for k in
                 ("clear_water", "muddy_water", "debris_flow", "large_rock")},
    } for t, row in rows]

    return {"camera": camera, "points": points, "truncated": truncated}


# Event types the current detector actually emits. camera_circuit_open,
# cooldown_skip and download_invalid appear in older log data but the present
# version never writes them, so they are not treated as live signals.
_ALERT_EVENTS = {"alert_change"}
_FAILURE_EVENTS = {"download_fail", "decode_fail", "no_image", "no_image_element", "exception"}


@app.get("/api/events")
def get_events(limit: int = 200, tail_bytes: int = 2_000_000):
    """
    Alert transitions, plus the most recent fetch failure per camera.

    events.jsonl is append-only and already ~9.5 MB, dominated by per-frame
    records, so only the tail is read. Failures are collapsed to one per camera:
    the dashboard wants to know 'is this camera's feed broken right now', not to
    receive two thousand individual failures.
    """
    result = {"events": [], "failures": {}, "available": EVENTS_FILE.exists()}
    if not EVENTS_FILE.exists():
        return result

    try:
        size = EVENTS_FILE.stat().st_size
        with open(EVENTS_FILE, "rb") as f:
            start = max(0, size - max(1024, tail_bytes))
            f.seek(start)
            blob = f.read()
    except OSError as e:
        logger.error(f"events read failed: {e}")
        return result

    lines = blob.decode("utf-8", errors="replace").split("\n")
    if start > 0 and lines:
        lines = lines[1:]         # first line is probably cut in half

    alerts = []
    failures = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue              # the file already contains a few damaged lines
        if not isinstance(rec, dict):
            continue
        ev, cam, ts = rec.get("event"), rec.get("camera"), rec.get("ts")
        if not ev or not ts:
            continue
        if ev in _ALERT_EVENTS:
            alerts.append({
                "ts": ts, "camera": cam,
                "prev": rec.get("prev"), "curr": rec.get("curr"),
                "yellowReason": rec.get("yellowReason") or "",
                "redReason": rec.get("redReason") or "",
            })
        elif ev in _FAILURE_EVENTS and cam:
            failures[cam] = {"ts": ts, "event": ev}

    alerts.reverse()              # newest first
    result["events"] = alerts[:max(1, min(limit, 1000))]
    result["failures"] = failures
    result["truncated"] = start > 0
    return result


# ===== ACKNOWLEDGEMENT ROUTES =====

def _read_acks() -> dict:
    """
    Replay acks.jsonl into the current state per camera.

    The file is append-only: an un-acknowledge is a new record, not a deletion,
    because this doubles as the record of who handled which alert and why.
    """
    state = {}
    if not ACKS_FILE.exists():
        return state
    try:
        with open(ACKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                cam = rec.get("camera")
                if not cam:
                    continue
                if rec.get("action") == "unack":
                    state.pop(cam, None)
                else:
                    state[cam] = rec
    except OSError as e:
        logger.error(f"acks read failed: {e}")
    return state


@app.get("/api/acks")
def get_acks():
    """Current acknowledgement per camera."""
    return {"acks": _read_acks()}


@app.post("/api/acks")
def post_ack(payload: dict = Body(...)):
    """
    Record an acknowledgement, an un-acknowledge, or an early release of a held
    red alert.

    Note what this does NOT do: it never clears the detector's 24-hour red hold.
    The detector reads its state file only at startup, so writing there would
    take effect at some unpredictable future restart, and a web click that
    silently disarms a safety mechanism is the wrong shape anyway. 'clear_red'
    records an operator decision that the console honours in its presentation;
    the detector keeps holding.
    """
    def field(name, maxlen, required=True):
        v = payload.get(name)
        v = "" if v is None else str(v).strip()
        if required and not v:
            raise HTTPException(status_code=400, detail=f"{name} is required")
        return v[:maxlen]

    action = field("action", 20)
    if action not in ACK_ACTIONS:
        raise HTTPException(status_code=400, detail="Unknown action")

    camera = field("camera", 120)
    by = field("by", 60)

    reason = field("reason", 40, required=(action != "unack"))
    if action != "unack":
        if reason not in REASON_CODES:
            raise HTTPException(status_code=400, detail="Unknown reason code")
        reason_text = field("reasonText", 500, required=(reason == "other"))
    else:
        reason, reason_text = "", ""

    images = payload.get("images") or {}
    if not isinstance(images, dict):
        images = {}

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "camera": camera,
        "action": action,
        "alert": field("alert", 20, required=False),
        "alertTs": field("alertTs", 40, required=False),
        # Scopes the record to one specific alert episode. For red this is
        # redSince, so a NEW debris flow at the same camera is never silenced by
        # an acknowledgement of the previous one.
        "eventKey": field("eventKey", 40, required=False),
        "by": by,
        # 'self' means the name was typed by the user and is not verified. If an
        # authenticating proxy is ever put in front of this, that becomes
        # 'access' and the record format does not have to change.
        "bySource": "self",
        "reason": reason,
        "reasonText": reason_text,
        # Kept for reason codes in the false-positive family: these frames are
        # labelled training data for whoever next tunes the model.
        "images": {
            "annotated": str(images.get("annotated") or "")[:500],
            "raw": str(images.get("raw") or "")[:500],
        },
        "notified": None,   # reserved for LINE push
    }

    try:
        with acks_lock:
            ACKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(ACKS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"ack write failed: {e}")
        raise HTTPException(status_code=500, detail="Cannot record acknowledgement")

    logger.info(f"ack {action} {camera} by {by} ({reason})")
    return {"success": True, "record": record}


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
    """Serve the operations console."""
    console = WEB_GIS_DIR / "console.html"
    if console.exists():
        return FileResponse(console, media_type="text/html")
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
