"""
Sync client for uploading detection results to cloud server
Run alongside yolo11m_seg_detectfromcctv_V3.PY to sync results to Render

Features:
- Automatic retry on failure
- Connection error handling
- Upload progress tracking
- Efficient image sync (latest per camera)
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("[WARN] watchdog not installed - file watching disabled")

# Load .env or secrets.env manually for local development
for env_file in [".env", "secrets.env"]:
    env_path = Path(__file__).parent / env_file
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        key, val = line.strip().split("=", 1)
                        if key not in os.environ:
                            os.environ[key] = val
        except Exception as e:
            print(f"[WARN] Failed to load {env_file}: {e}")

RENDER_URL = os.environ.get("RENDER_URL", "https://debris-flow-monitoring.onrender.com")
API_KEY = os.environ.get("SYNC_API_KEY")

if not API_KEY:
    print("[WARN] SYNC_API_KEY not found in environment or .env")

# Local paths
SCRIPT_DIR = Path(__file__).parent.resolve()
CAMERA_STATUS_FILE = SCRIPT_DIR / "camera_status.json"
SAVED_IMAGES_DIR = SCRIPT_DIR / "saved_images"

# Sync settings
SYNC_INTERVAL = 5  # seconds between periodic syncs
SYNC_IMAGES = True  # Whether to sync images
MAX_RETRIES = 3  # Max retry attempts
RETRY_DELAY = 2  # Seconds between retries
CONNECTION_TIMEOUT = 30  # Request timeout
UPLOAD_TIMEOUT = 60  # Image upload timeout

# ============ SESSION SETUP WITH RETRY ============

def create_session() -> requests.Session:
    """Create a requests session with retry logic"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session
session = create_session()

# ============ STATS TRACKING ============

class SyncStats:
    """Track sync statistics"""
    def __init__(self):
        self.status_syncs = 0
        self.image_syncs = 0
        self.errors = 0
        self.last_sync: Optional[str] = None
        self.synced_images: Set[str] = set()  # Track already synced images
    
    def log_sync(self, sync_type: str):
        self.last_sync = datetime.now().strftime("%H:%M:%S")
        if sync_type == "status":
            self.status_syncs += 1
        elif sync_type == "image":
            self.image_syncs += 1
    
    def log_error(self):
        self.errors += 1
    
    def summary(self) -> str:
        return f"Status: {self.status_syncs} | Images: {self.image_syncs} | Errors: {self.errors} | Last: {self.last_sync or 'Never'}"

stats = SyncStats()

# ============ SYNC FUNCTIONS ============

def check_server_health() -> bool:
    """Check if the server is reachable"""
    try:
        response = session.get(
            f"{RENDER_URL}/api/health",
            timeout=CONNECTION_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            print(f"[HEALTH] Server OK - Cameras: {data.get('cameras', 0)} | Key: {'✓' if data.get('api_key_set') else '✗'}")
            return True
        else:
            print(f"[HEALTH] Server returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[HEALTH] Cannot connect to server - check internet connection")
        return False
    except Exception as e:
        print(f"[HEALTH] Error: {e}")
        return False

def rewrite_urls(data: Dict) -> Dict:
    """Rewrite localhost URLs to relative paths"""
    for cam, info in data.items():
        # Fix main image URLs
        for key in ['imageUrl', 'rawUrl']:
            if key in info and info[key]:
                url = info[key]
                if '/saved_images/' in url:
                    info[key] = '/saved_images/' + url.split('/saved_images/', 1)[1]
        
        # Fix history URLs
        for key in ['history', 'historyRaw']:
            if key in info and isinstance(info[key], list):
                info[key] = [
                    '/saved_images/' + u.split('/saved_images/', 1)[1] 
                    if '/saved_images/' in u else u 
                    for u in info[key]
                ]
    return data

def sync_status() -> bool:
    """Upload camera_status.json to cloud"""
    if not CAMERA_STATUS_FILE.exists():
        print(f"[SYNC] No camera_status.json found")
        return False
    
    try:
        with open(CAMERA_STATUS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Rewrite URLs
        data = rewrite_urls(data)
        
        response = session.post(
            f"{RENDER_URL}/api/sync/status",
            headers={"X-API-Key": API_KEY},
            data={"status_data": json.dumps(data, ensure_ascii=False)},
            timeout=CONNECTION_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            stats.log_sync("status")
            print(f"[SYNC] Status: {result.get('cameras', 0)} cameras | {stats.summary()}")
            return True
        else:
            stats.log_error()
            print(f"[SYNC] Status failed: {response.status_code} - {response.text[:100]}")
            return False
            
    except requests.exceptions.ConnectionError:
        stats.log_error()
        print("[SYNC] Connection error - server unreachable")
        return False
    except json.JSONDecodeError as e:
        stats.log_error()
        print(f"[SYNC] JSON error: {e}")
        return False
    except Exception as e:
        stats.log_error()
        print(f"[SYNC] Status error: {e}")
        return False

def sync_image(cam_name: str, filename: str, filepath: Path) -> bool:
    """Upload a single image to cloud with retry"""
    image_key = f"{cam_name}/{filename}"
    
    # Skip if already synced this session
    if image_key in stats.synced_images:
        return True
    
    for attempt in range(MAX_RETRIES):
        try:
            with open(filepath, 'rb') as f:
                response = session.post(
                    f"{RENDER_URL}/api/sync/image",
                    headers={"X-API-Key": API_KEY},
                    data={"camera": cam_name, "filename": filename},
                    files={"image": (filename, f, "image/jpeg")},
                    timeout=UPLOAD_TIMEOUT
                )
            
            if response.status_code == 200:
                stats.log_sync("image")
                stats.synced_images.add(image_key)
                return True
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                stats.log_error()
                print(f"[SYNC] Image failed: {cam_name}/{filename} - {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            stats.log_error()
            print(f"[SYNC] Image connection error: {cam_name}/{filename}")
            return False
        except Exception as e:
            stats.log_error()
            print(f"[SYNC] Image error: {e}")
            return False
    
    return False

# ============ THREAD POOL ============
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create a thread pool for parallel uploads
executor = ThreadPoolExecutor(max_workers=8)

def sync_recent_images() -> int:
    """Sync the LATEST image from each camera folder in PARALLEL"""
    if not SAVED_IMAGES_DIR.exists():
        return 0
    
    synced = 0
    total_cams = 0
    
    # Collect all upload tasks
    upload_tasks = []
    
    for cam_dir in SAVED_IMAGES_DIR.iterdir():
        if not cam_dir.is_dir() or cam_dir.name == 'demo':
            continue
        
        total_cams += 1
        cam_name = cam_dir.name
        
        # Find the most recent image
        jpg_files = list(cam_dir.glob("*.jpg"))
        if not jpg_files:
            continue
        
        latest_file = max(jpg_files, key=lambda f: f.stat().st_mtime)
        
        # Submit task to executor
        upload_tasks.append(
            executor.submit(sync_image, cam_name, latest_file.name, latest_file)
        )
    
    # Process results as they complete
    if upload_tasks:
        for future in as_completed(upload_tasks):
            try:
                if future.result():
                    synced += 1
            except Exception as e:
                print(f"[SYNC] Parallel task error: {e}")
    
    if synced > 0:
        print(f"[SYNC] Images: {synced}/{total_cams} cameras (Parallel) | {stats.summary()}")
    
    return synced

# ============ FILE WATCHER ============

if WATCHDOG_AVAILABLE:
    class StatusFileHandler(FileSystemEventHandler):
        """Watch for changes to camera_status.json"""
        def __init__(self):
            self.last_sync = 0
        
        def on_modified(self, event):
            if event.src_path.endswith("camera_status.json"):
                now = time.time()
                if now - self.last_sync > 2:  # Debounce
                    self.last_sync = now
                    # Upload images FIRST, then status
                    if SYNC_IMAGES:
                        sync_recent_images()
                    sync_status()

# ============ MAIN ============

def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 60)
    print("   🌊 Debris Flow Cloud Sync Client v2.0")
    print("=" * 60)
    print(f"   Server:  {RENDER_URL}")
    print(f"   Status:  {CAMERA_STATUS_FILE}")
    print(f"   Images:  {SAVED_IMAGES_DIR}")
    print(f"   Sync:    Every {SYNC_INTERVAL}s | Images: {'ON' if SYNC_IMAGES else 'OFF'}")
    print(f"   Retry:   {MAX_RETRIES} attempts with {RETRY_DELAY}s delay")
    print("=" * 60 + "\n")

def main():
    print_banner()
    
    # Validate API key
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  ERROR: API key not configured!")
        print("   Set SYNC_API_KEY environment variable or edit this file.")
        sys.exit(1)
    
    # Check server health
    print("[INIT] Checking server connection...")
    if not check_server_health():
        print("[WARN] Server unreachable - will retry during sync")
    
    # Initial sync
    print("\n[INIT] Performing initial sync...")
    if SYNC_IMAGES:
        sync_recent_images()
    sync_status()
    
    # Setup file watcher if available
    observer = None
    if WATCHDOG_AVAILABLE:
        event_handler = StatusFileHandler()
        observer = Observer()
        observer.schedule(event_handler, str(SCRIPT_DIR), recursive=False)
        observer.start()
        print("[WATCH] File watcher started")
    else:
        print("[WATCH] File watcher disabled (install watchdog for real-time sync)")
    
    print(f"\n[RUN] Syncing every {SYNC_INTERVAL}s (Ctrl+C to stop)...\n")
    
    try:
        while True:
            time.sleep(SYNC_INTERVAL)
            # Periodic sync as backup
            if SYNC_IMAGES:
                sync_recent_images()
            sync_status()
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Shutting down...")
        print(f"[STATS] Final: {stats.summary()}")
        if observer:
            observer.stop()
            observer.join()

if __name__ == "__main__":
    main()
