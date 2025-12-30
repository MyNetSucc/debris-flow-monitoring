"""
Sync client for uploading detection results to cloud server
Run alongside yolo11m_seg_detectfromcctv_V3.PY to sync results to Render
"""
import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============ CONFIGURATION ============
# Set these or use environment variables
RENDER_URL = os.environ.get("RENDER_URL", "https://debris-flow-monitoring.onrender.com")
API_KEY = os.environ.get("SYNC_API_KEY", "jDYu96zua76zAQ2USgbFDMkicCnlUiAJzfk8xG_HdeI")  # Set this!

# Local paths
SCRIPT_DIR = Path(__file__).parent.resolve()
CAMERA_STATUS_FILE = SCRIPT_DIR / "camera_status.json"
SAVED_IMAGES_DIR = SCRIPT_DIR / "saved_images"

# Sync settings
SYNC_INTERVAL = 5  # seconds between status syncs
SYNC_IMAGES = True  # Whether to sync images (uses bandwidth)
MAX_IMAGE_AGE = 60  # Only sync images newer than this (seconds)

# ============ SYNC FUNCTIONS ============

def sync_status():
    """Upload camera_status.json to cloud"""
    if not CAMERA_STATUS_FILE.exists():
        print(f"[SYNC] No camera_status.json found")
        return False
    
    try:
        with open(CAMERA_STATUS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Rewrite image URLs from localhost to relative paths
        for cam, info in data.items():
            for key in ['imageUrl', 'rawUrl']:
                if key in info and info[key]:
                    url = info[key]
                    # Convert http://127.0.0.1:8000/saved_images/... to /saved_images/...
                    if '/saved_images/' in url:
                        info[key] = '/saved_images/' + url.split('/saved_images/', 1)[1]
            # Also fix history URLs
            for key in ['history', 'historyRaw']:
                if key in info and isinstance(info[key], list):
                    info[key] = [
                        '/saved_images/' + u.split('/saved_images/', 1)[1] 
                        if '/saved_images/' in u else u 
                        for u in info[key]
                    ]
        
        response = requests.post(
            f"{RENDER_URL}/api/sync/status",
            headers={"X-API-Key": API_KEY},
            data={"status_data": json.dumps(data, ensure_ascii=False)},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[SYNC] Status uploaded: {result.get('cameras', 0)} cameras")
            return True
        else:
            print(f"[SYNC] Status upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[SYNC] Status sync error: {e}")
        return False

def sync_image(cam_name: str, filename: str, filepath: Path):
    """Upload a single image to cloud"""
    try:
        with open(filepath, 'rb') as f:
            response = requests.post(
                f"{RENDER_URL}/api/sync/image",
                headers={"X-API-Key": API_KEY},
                data={"camera": cam_name, "filename": filename},
                files={"image": (filename, f, "image/jpeg")},
                timeout=60
            )
        
        if response.status_code == 200:
            print(f"[SYNC] Image uploaded: {cam_name}/{filename}")
            return True
        else:
            print(f"[SYNC] Image upload failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[SYNC] Image sync error: {e}")
        return False

def sync_recent_images():
    """Sync images modified in the last MAX_IMAGE_AGE seconds"""
    if not SAVED_IMAGES_DIR.exists():
        return
    
    now = time.time()
    synced = 0
    
    for cam_dir in SAVED_IMAGES_DIR.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name
        
        for img_file in cam_dir.glob("*.jpg"):
            age = now - img_file.stat().st_mtime
            if age < MAX_IMAGE_AGE:
                if sync_image(cam_name, img_file.name, img_file):
                    synced += 1
    
    if synced > 0:
        print(f"[SYNC] Synced {synced} recent images")

# ============ FILE WATCHER ============

class StatusFileHandler(FileSystemEventHandler):
    """Watch for changes to camera_status.json"""
    def __init__(self):
        self.last_sync = 0
    
    def on_modified(self, event):
        if event.src_path.endswith("camera_status.json"):
            now = time.time()
            if now - self.last_sync > 2:  # Debounce
                self.last_sync = now
                sync_status()
                if SYNC_IMAGES:
                    sync_recent_images()

# ============ MAIN ============

def main():
    print("=" * 50)
    print("Debris Flow Cloud Sync Client")
    print("=" * 50)
    print(f"Server: {RENDER_URL}")
    print(f"Status file: {CAMERA_STATUS_FILE}")
    print(f"Images dir: {SAVED_IMAGES_DIR}")
    print(f"Sync images: {SYNC_IMAGES}")
    print("=" * 50)
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  WARNING: API key not configured!")
        print("   Set SYNC_API_KEY environment variable or edit this file.\n")
    
    # Initial sync
    print("\n[SYNC] Performing initial sync...")
    sync_status()
    if SYNC_IMAGES:
        sync_recent_images()
    
    # Watch for changes
    event_handler = StatusFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(SCRIPT_DIR), recursive=False)
    observer.start()
    
    print(f"\n[SYNC] Watching for changes (Ctrl+C to stop)...")
    
    try:
        while True:
            time.sleep(SYNC_INTERVAL)
            # Periodic sync as backup
            sync_status()
    except KeyboardInterrupt:
        print("\n[SYNC] Stopping...")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
