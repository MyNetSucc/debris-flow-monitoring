#!/usr/bin/env python3
# yolo11m_seg_detectfromcctv_V3.py
# -------------------------------------------------
#   • 類別統一：clear_water / muddy_water / debris_flow / large_rock
#   • camera_status.json：proportion(比例0-1) / confidence(最大信心) / rawUrl / historyRaw
#   • 黃/紅燈邏輯：紅燈 hold 一天、黃燈退場、紅燈不被覆蓋
#   • Rock 以 IoU 追蹤連續位移判定移動
#   • 保留 history 影像 URL 串列（偵測圖 + 原圖；舊→新）
#   • 日誌：logs/app.log(輪替) / logs/events.jsonl / logs/metrics/<cam>.csv
#   • 效能：Selenium Driver 於 worker 內重用、requests.Session() 共用
#   • 抗 Timeout：自動掃頁面與 iframe 取第一個可用影像來源
#   • 動態頻率：紅/黃燈自動提高偵測頻率（最早到期排程）
#   • ★強化：支援 data:image;base64、page load timeout、history 留存可參數化
# -------------------------------------------------

import os, sys, time, json, threading, csv, logging, argparse, base64, re
from logging.handlers import RotatingFileHandler
from collections import deque, defaultdict, Counter
from datetime import datetime
import numpy as np
import cv2, requests, urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ReadTimeout, ConnectionError, SSLError
from ultralytics import YOLO

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by  import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support    import expected_conditions as EC
from selenium.common.exceptions    import TimeoutException, NoSuchElementException
from urllib.parse import quote, urljoin
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================== CLI 參數 ================== #
# 自動偵測腳本所在目錄，支援相對路徑
SCRIPT_DIR = Path(__file__).parent.resolve()

# 智慧型路徑解析：優先使用相對路徑，若不存在則嘗試絕對路徑
def resolve_path(relative_path, fallback_absolute=None):
    """解析路徑：優先使用相對於腳本的路徑"""
    rel = SCRIPT_DIR / relative_path
    if rel.exists():
        return str(rel)
    if fallback_absolute and Path(fallback_absolute).exists():
        return fallback_absolute
    return str(rel)  # 回傳相對路徑（即使不存在，讓後續報錯更清晰）

ap = argparse.ArgumentParser(
    description='Debris-flow realtime detector with dynamic frequency & robust image source resolver',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
範例:
  python %(prog)s                           # 使用預設設定
  python %(prog)s --focus 攝影機A 攝影機B    # 只監看特定攝影機
  python %(prog)s --interval 2.0            # 設定偵測間隔為 2 秒
  python %(prog)s --model weights/best.pt   # 指定模型路徑
'''
)
# YOLO / 行為
ap.add_argument('--model', default=resolve_path('weights/yolo11m-seg.pt'),
                help='YOLO 模型權重檔路徑 (預設: weights/yolo11m-seg.pt)')
ap.add_argument('--conf', type=float, default=0.539, help='YOLO 信心門檻 (預設: 0.539)')
ap.add_argument('--iou',  type=float, default=0.50, help='YOLO NMS IoU 門檻 (預設: 0.50)')
ap.add_argument('--site', type=str, default='debris_flow_monitoring', help='站點名稱，用於日誌與識別')
ap.add_argument('--no-frame-log', action='store_true', help='停用每幀日誌輸出到 app.log')

# 程式 / 環境
ap.add_argument('--focus', nargs='*', default=[], metavar='CAM',
                help='只監看指定攝影機 (空=監看全部)，可指定多個')
ap.add_argument('--threads', type=int, default=5, help='並行 worker 執行緒數 (預設: 5)')
ap.add_argument('--interval', type=float, default=1.0, help='基本偵測間隔秒數 (預設: 1.0)')
ap.add_argument('--interval-green',  type=float, default=None, metavar='SEC', help='綠燈偵測間隔秒（預設用 --interval）')
ap.add_argument('--interval-yellow', type=float, default=None, metavar='SEC', help='黃燈偵測間隔秒（預設 0.75 * --interval）')
ap.add_argument('--interval-red',    type=float, default=None, metavar='SEC', help='紅燈偵測間隔秒（預設 0.50 * --interval）')
ap.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                help='日誌等級 (預設: INFO)')
ap.add_argument('--headless', action='store_true', default=True, help='以無頭模式執行 Chrome (預設: 開啟)')
ap.add_argument('--chromedriver', default=resolve_path('chromedriver-win64/chromedriver.exe'),
                help='ChromeDriver 路徑 (預設: chromedriver-win64/chromedriver.exe)')

# I/O
ap.add_argument('--save-root', default=str(SCRIPT_DIR / 'saved_images'),
                help='影像儲存根目錄 (預設: saved_images/)')
ap.add_argument('--max-storage', type=float, default=100 * 1024**3, metavar='BYTES',
                help='最大儲存空間 bytes (預設: 100GB)')
ap.add_argument('--status-json', default='camera_status.json',
                help='攝影機狀態輸出 JSON 檔 (預設: camera_status.json)')
ap.add_argument('--cctv-json',   default=str(SCRIPT_DIR / 'cctv.json'),
                help='攝影機清單 JSON 檔 (預設: cctv.json)')
ap.add_argument('--base-url',    default=os.getenv('BASE_URL', 'http://127.0.0.1:8000'),
                help='（保留參數，已不再寫入 camera_status.json；影像一律存相對路徑以利跨主機瀏覽）')

# ★ 可參數化的 history 留存規則（預設與原本一致）
ap.add_argument('--history-keep-sec', type=float, default=30.0, help='history 保留秒數（預設 30）')
ap.add_argument('--history-max',      type=int,   default=40,   help='history 最大張數（預設 40）')

args = ap.parse_args()

# ================== 參數與常數 ================== #
CONF_THRES     = float(args.conf)
IOU_THRES      = float(args.iou)
SITE_NAME      = args.site
LOG_EVERY_FRAME= not args.no_frame_log

FOCUS_LIST   = args.focus
THREADS      = max(1, int(args.threads))
SAVE_ROOT    = args.save_root
MAX_STORAGE  = float(args.max_storage)
MODEL_PATH   = args.model
CHROMEDRIVER = args.chromedriver
CAM_STATUS_JS= args.status_json
CCTV_JSON    = args.cctv_json
BASE_URL     = args.base_url
INTERVAL_SEC = max(0.2, float(args.interval))

INTERVAL_GREEN  = max(0.2, float(args.interval_green))  if args.interval_green  is not None else INTERVAL_SEC
INTERVAL_YELLOW = max(0.2, float(args.interval_yellow)) if args.interval_yellow is not None else max(0.2, INTERVAL_SEC * 0.75)
INTERVAL_RED    = max(0.2, float(args.interval_red))    if args.interval_red    is not None else max(0.2, INTERVAL_SEC * 0.50)

# ★ history 留存參數
HISTORY_KEEP_SEC = float(args.history_keep_sec)
HISTORY_MAX      = int(args.history_max)

# ── 追蹤與配對（rock）──────────────────────
IOU_TH_ROCK        = 0.30
MOVE_CONSEC        = 3
# ── 判定門檻 ─────────────────────────────
BASE_CONF          = 0.539   # 基礎信心門檻
MIN_PROP           = 0.0008  # 最小比例門檻（0.08%）
# ── 黃/紅燈時間 ─────────────────────────
HOLD_RED_SEC       = 86400   # 紅燈 hold 1 天
MIN_YELLOW_SEC     = 8
YELLOW_CLEAR_SEC   = 90

# ================== 類別與外觀 ================== #
LABELS = ['clear_water', 'muddy_water', 'debris_flow', 'large_rock']
CLASS_MAP = {
    'clear_water': 'clear_water',
    'muddy_water': 'muddy_water',
    'debris_flow': 'debris_flow',
    'large_rock' : 'large_rock',
    # 舊名兼容
    'river'      : 'clear_water',
    'riverbond'  : 'clear_water',
    'muddy'      : 'muddy_water',
    'debris-flow': 'debris_flow',
    'rock'       : 'large_rock',
}
COLORMAP = {
    'debris_flow': (0, 0, 255),
    'muddy_water': (255, 0, 0),
    'clear_water': (0, 255, 255),
    'large_rock' : (0, 255, 0)
}

# ================== Logging ================== #
LOG_DIR = Path('logs'); LOG_DIR.mkdir(parents=True, exist_ok=True)
app_handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10_000_000, backupCount=5, encoding='utf-8')
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s',
    handlers=[app_handler, logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('df-app')

EVENTS_JSONL = str(LOG_DIR / 'events.jsonl')
METRICS_DIR  = 'logs/metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

def log_event(event: str, cam: str, **payload):
    rec = {'ts': datetime.utcnow().isoformat()+'Z', 'event': event, 'camera': cam, 'site': SITE_NAME}
    rec.update(payload)
    with open(EVENTS_JSONL, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def append_metrics(cam: str, timestamp: str, prop: dict, conf: dict, alert: str, y_reason: str, r_reason: str):
    path = os.path.join(METRICS_DIR, f'{cam}.csv')
    hdr = ['timestamp'] + [f'{k}_prop' for k in LABELS] + [f'{k}_conf' for k in LABELS] + ['alert','yellowReason','redReason']
    newfile = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if newfile: w.writerow(hdr)
        row = [timestamp] + [prop.get(k,0) for k in LABELS] + [conf.get(k,0) for k in LABELS] + [alert, y_reason, r_reason]
        w.writerow(row)

# ================== YOLO & HTTP ================== #
model = YOLO(MODEL_PATH)
model.conf = CONF_THRES
model.iou  = IOU_THRES

session = requests.Session()
retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
adapter = HTTPAdapter(max_retries=retry, pool_connections=THREADS*4, pool_maxsize=THREADS*8)
session.mount('http://', adapter)
session.mount('https://', adapter)

def download(url, timeout=20):
    try:
        r = session.get(url, timeout=timeout, verify=False)
        if r.status_code == 200:
            return r
    except (ReadTimeout, ConnectionError, SSLError):
        pass
    return None

# ★ 支援 data:image/*;base64 與 http(s)
DATA_URL_RE = re.compile(r'^data:image/(png|jpeg|jpg);base64,', re.I)

def read_image_bytes(src: str, timeout=20):
    """回傳位元組：支援 http(s) 與 data:image/base64"""
    if not src:
        return None
    m = DATA_URL_RE.match(src.strip())
    if m:
        try:
            b64 = DATA_URL_RE.sub('', src.strip())
            return base64.b64decode(b64)
        except Exception:
            return None
    r = download(src, timeout=timeout)
    return r.content if r is not None else None

_storage_lock  = threading.Lock()
_storage_state = {'total': None, 'checked_ts': 0.0}
STORAGE_RESCAN_SEC = 300   # 沒有超標時，最多每 5 分鐘才重新掃描一次

def note_new_bytes(n: int):
    """把剛寫出的影像大小累加進快取總量，讓快取值不會隨時間低估。"""
    with _storage_lock:
        if _storage_state['total'] is not None:
            _storage_state['total'] += n

def keep_under(folder, limit):
    """
    ★ 效能：原本每偵測一張影像就完整走訪整個 saved_images/ 加總大小，
      而且每刪掉一個檔案就再全樹重掃一次（O(n²)）。63 台攝影機下等於每秒
      數十次全樹掃描，檔案一多就成為整個流程的主要瓶頸。
      改為：快取總量，僅在「尚未掃描 / 逾時 / 已超標」時才重新掃描，
      且一次掃描就完成排序與刪除。
    """
    now = time.time()
    with _storage_lock:
        total = _storage_state['total']
        due = (total is None
               or (now - _storage_state['checked_ts']) > STORAGE_RESCAN_SEC
               or total > limit)
    if not due:
        return

    files = []
    total = 0
    for dp, _, fs in os.walk(folder):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))
            total += st.st_size

    if total > limit and files:
        files.sort()  # 最舊的先刪
        for _, size, p in files:
            if total <= limit:
                break
            try:
                os.remove(p)
                total -= size
                log.warning(f"Storage limit trimming: removed {p}")
            except OSError as e:
                log.error(f"Remove file failed: {e}")

    with _storage_lock:
        _storage_state['total'] = total
        _storage_state['checked_ts'] = time.time()

def iou_box(a,b):
    xA, yA = max(a[0],b[0]), max(a[1],b[1])
    xB, yB = min(a[2],b[2]), min(a[3],b[3])
    inter  = max(0,xB-xA) * max(0,yB-yA)
    areaA  = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-6)

def log_frame_summary(cam_name, res, frame_idx=None, frame_ms=None):
    names_src = getattr(res, 'names', None) or model.names
    labels = []
    for b in getattr(res, 'boxes', []):
        raw = names_src[int(b.cls[0])]
        lbl = CLASS_MAP.get(raw, raw)
        labels.append(lbl)
    has_masks = getattr(res, 'masks', None) is not None
    counts = dict(Counter(labels))
    pos_str = None
    if frame_ms is not None and frame_ms > 0:
        total_ms = int(frame_ms)
        sec, ms = divmod(total_ms, 1000)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        pos_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    log.info(
        f"site={SITE_NAME} cam={cam_name} time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"frame_idx={frame_idx} video_pos={pos_str} has_masks={has_masks} conf={CONF_THRES} classes={counts}"
    )

# ================== 全域狀態 ================== #
hist_queues_det      = defaultdict(deque)     # cam -> deque[(ts,url)] 偵測圖（舊→新）
hist_queues_raw      = defaultdict(deque)     # cam -> deque[(ts,url)] 原圖（舊→新）
prev_rock_boxes      = defaultdict(deque)     # cam -> deque[[x1,y1,x2,y2],...]*N
prev_move_cnt        = defaultdict(int)       # cam -> 連續移動幀數
alert_state          = defaultdict(lambda: ('green', 0.0))   # cam -> (level, expire_ts)
red_since_ts         = defaultdict(float)     # cam -> 本次紅燈起始時間（0 = 非紅燈）
report_ts            = defaultdict(deque)     # cam -> 最近成功回報時間，用來估回報節奏
yellow_since_ts      = defaultdict(float)     # cam -> 黃燈起始時間
last_muddy_seen_ts   = defaultdict(float)     # cam -> 最近看到 muddy 時間
camera_status        = {}                     # 最新 camera_status.json 快取
status_lock          = threading.Lock()       # camera_status 讀寫與檔案寫出的互斥鎖
state_lock           = threading.Lock()       # alert_state 落地檔的互斥鎖

# ── 警戒狀態持久化 ──────────────────────────────────────────────
# alert_state 原本只活在記憶體裡，程序一重啟，HOLD_RED_SEC 的 24 小時紅燈鎖定
# 就無聲消失——重開機、改參數、當機重跑都會。判定邏輯看起來仍然正常，實際上
# 保護已經不見了。這裡把它落地，啟動時讀回。
# expire_ts 存的是 time.time() 的絕對 epoch 秒，重啟後仍可直接比較。
ALERT_STATE_FILE = str(Path(CAM_STATUS_JS).with_name('alert_state.json'))

def save_alert_state():
    """寫出 alert_state（原子置換）。只在燈號真的改變時呼叫，不是每幀。"""
    with state_lock:
        snapshot = {
            'saved_at': time.time(),
            'cameras': {
                cam: {'level': lvl, 'expire': exp, 'redSince': red_since_ts.get(cam, 0.0)}
                for cam, (lvl, exp) in list(alert_state.items())
                if lvl == 'red' and exp
            }
        }
        tmp = ALERT_STATE_FILE + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            os.replace(tmp, ALERT_STATE_FILE)
        except OSError as e:
            log.error(f'寫出 {ALERT_STATE_FILE} 失敗: {e}')
            try: os.remove(tmp)
            except OSError: pass

def load_alert_state():
    """
    啟動時讀回仍在鎖定中的紅燈。

    ★ 這個函式必須 fail-open：檔案不存在、JSON 壞掉、欄位型別怪異，一律當作
      「沒有前次狀態」繼續啟動。沒有這個檔案時系統本來就能正常運作，讓它變成
      啟動的必要條件，等於拿可用性去換一致性，方向是錯的。
    """
    try:
        with open(ALERT_STATE_FILE, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except Exception as e:
        log.warning(f'{ALERT_STATE_FILE} 無法讀取（{e}），以空狀態啟動')
        return

    now = time.time()
    restored = 0
    for cam, rec in (data.get('cameras') or {}).items():
        try:
            if str(rec.get('level')) != 'red':
                continue
            exp = float(rec.get('expire') or 0.0)
            if exp <= now:
                continue          # 鎖定已到期，讓它重新判定
            alert_state[cam] = ('red', exp)
            red_since_ts[cam] = float(rec.get('redSince') or 0.0)
            restored += 1
        except (TypeError, ValueError):
            continue              # 單筆壞掉就跳過，不影響其他攝影機
    if restored:
        log.info(f'自 {ALERT_STATE_FILE} 回復 {restored} 個仍在鎖定中的紅燈')

def note_report(cam: str, ts: float) -> float:
    """
    記錄一次成功回報，回傳近期回報間隔的中位數（秒）。

    網頁端用它判斷某站是不是「回報變慢／沒回報」。寫死秒數會在 --threads 或
    --interval 一改就失準，所以基準線由知道真相的偵測端提供，門檻倍數留在前端。
    取中位數而非平均，是因為偶爾一次逾時重試會把平均拉歪。
    """
    q = report_ts[cam]
    q.append(ts)
    while len(q) > 21:
        q.popleft()
    if len(q) < 3:
        return 0.0
    lst = list(q)
    gaps = sorted(b - a for a, b in zip(lst, lst[1:]))
    n = len(gaps)
    med = gaps[n // 2] if n % 2 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2
    return round(med, 2)

def write_camera_status(cam: str, entry: dict):
    """
    ★ 執行緒安全地更新並寫出 camera_status.json。
      原本每個 worker 直接 open(...,'w') 覆寫同一個檔案且沒有任何鎖，
      兩條執行緒同時寫入會互相截斷，產生半截 / 壞掉的 JSON；
      讀取端（server.py、sync_to_cloud.py）就會整批漏掉更新。
      先寫暫存檔再 os.replace（原子性 rename），讀取端不會再看到寫到一半的檔案。
    """
    with status_lock:
        camera_status[cam] = entry
        # 在鎖內先序列化：json.dump 會邊走訪 dict 邊輸出，若此時別的執行緒插入新
        # 攝影機，原本會拋 RuntimeError: dictionary changed size during iteration，
        # 被 detect_once 的 except 吞掉 → 該次偵測結果就這樣無聲遺失。
        payload = json.dumps(camera_status, ensure_ascii=False, indent=2)
        tmp = f'{CAM_STATUS_JS}.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(payload)
            # Windows 上若讀取端正開著目標檔，rename 會 PermissionError。
            # 逐次加長退避（最壞約 1.1 秒）；讀取端同樣有重試，兩邊夾擊下這個
            # 視窗會很快關閉。真的重試到底就放棄這一次更新（下一輪會補上），
            # 總比寫出半截檔案讓所有讀取端拿到壞資料好。
            for attempt in range(10):
                try:
                    os.replace(tmp, CAM_STATUS_JS)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        except OSError as e:
            log.error(f'寫出 {CAM_STATUS_JS} 失敗: {e}')
            try: os.remove(tmp)
            except OSError: pass

# ================== Selenium ================== #
def _chrome_options(headless=True):
    opt = Options()
    if headless:
        opt.add_argument('--headless=new'); opt.add_argument('--disable-gpu')
    opt.add_argument('--no-sandbox')
    opt.add_argument('--disable-dev-shm-usage')
    opt.add_argument('--window-size=1920,1080')
    # 加速：不載入圖片（我們只拿 src）
    prefs = {"profile.managed_default_content_settings.images": 2}
    opt.add_experimental_option("prefs", prefs)
    opt.page_load_strategy = 'eager'
    # 放寬混合內容/私網政策（常見環境更穩）
    opt.add_argument('--allow-running-insecure-content')
    opt.add_argument('--disable-features=BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessChecks')
    return opt

def create_driver(headless=True):
    """
    優先使用隨附的 chromedriver，失敗時改由 Selenium Manager 自動取得相符版本。
    ★ Chrome 會自動更新，隨附的 chromedriver 不會；版本一旦不符，原本會讓所有
      worker 在啟動當下就拋例外而全數死亡（且主迴圈不會察覺）。
    """
    last_err = None
    if CHROMEDRIVER and Path(CHROMEDRIVER).exists():
        try:
            driver = webdriver.Chrome(service=Service(CHROMEDRIVER), options=_chrome_options(headless))
            driver.set_page_load_timeout(25)
            return driver
        except Exception as e:
            # 版本不符會是 SessionNotCreatedException，但檔案缺失/無法執行時是
            # FileNotFoundError(OSError)，不屬於 WebDriverException，因此這裡放寬
            # 攔截範圍——反正後面還有 Selenium Manager 這條退路。
            last_err = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
            log.warning(f'隨附的 chromedriver 無法使用（{type(e).__name__}: {last_err}），改用 Selenium Manager 自動解析驅動程式')

    try:
        # 未指定 Service 路徑時，Selenium (>=4.6) 會自動下載/選用相符的 driver
        driver = webdriver.Chrome(options=_chrome_options(headless))
        driver.set_page_load_timeout(25)
        return driver
    except Exception as e:
        raise RuntimeError(
            f'無法啟動 Chrome。隨附 driver 錯誤: {last_err}；Selenium Manager 錯誤: {e}'
        ) from e

def _scan_for_media_src(driver):
    """在當前 browsing context 掃描可能的影像來源並回傳第一個非空 src。"""
    for css in ["img#img1", "img#Image1", "img#camimg"]:
        for e in driver.find_elements(By.CSS_SELECTOR, css):
            src = (e.get_attribute('src') or '').strip()
            if src: return src
    for e in driver.find_elements(By.TAG_NAME, 'img'):
        src = (e.get_attribute('src') or '').strip()
        if src: return src
    for v in driver.find_elements(By.TAG_NAME, 'video'):
        src = (v.get_attribute('src') or '').strip()
        if src: return src
        for s in v.find_elements(By.TAG_NAME, 'source'):
            ss = (s.get_attribute('src') or '').strip()
            if ss: return ss
    for c in driver.find_elements(By.CSS_SELECTOR, 'canvas[data-url], canvas[data-src]'):
        src = (c.get_attribute('data-url') or c.get_attribute('data-src') or '').strip()
        if src: return src
    return None

def resolve_image_src(driver, page_url: str, wait_sec: float = 25.0) -> str | None:
    """在頁面與所有 iframe/frames 中找出一個可下載的影像來源 URL。回傳絕對網址，或 None。"""
    t_end = time.time() + wait_sec
    last_err = None
    while time.time() < t_end:
        try:
            driver.switch_to.default_content()
            src = _scan_for_media_src(driver)
            if src:
                return urljoin(page_url, src)
            frames = driver.find_elements(By.CSS_SELECTOR, 'iframe, frame')
            for i in range(len(frames)):
                try:
                    driver.switch_to.default_content()
                    frames = driver.find_elements(By.CSS_SELECTOR, 'iframe, frame')
                    driver.switch_to.frame(frames[i])
                    src = _scan_for_media_src(driver)
                    if src:
                        driver.switch_to.default_content()
                        return urljoin(page_url, src)
                except Exception as e:
                    last_err = e
                    driver.switch_to.default_content()
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    if last_err:
        log.debug(f"resolve_image_src last error: {last_err}")
    return None

# ================== 動態間隔 ================== #
def _desired_interval_for(cam: str) -> float:
    s = camera_status.get(cam, {})
    alert = s.get('alert') or alert_state.get(cam, ('green', 0.0))[0]
    if alert == 'red':    return INTERVAL_RED
    if alert == 'yellow': return INTERVAL_YELLOW
    return INTERVAL_GREEN

# ================== 主偵測 ================== #
def detect_once(cam, url, driver: webdriver.Chrome):
    try:
        driver.get(url)
        src = resolve_image_src(driver, url, wait_sec=25)

        if not src:
            log_event('no_image_element', cam, page=url)
            log.warning(f"{cam} 找不到可用影像元素或 src（頁面/iframe 皆無）")
            return

        if 'noCCD_2.jpg' in src:
            log_event('no_image', cam, page=url, src=src)
            return

        # ★ 下載或解 data:image;base64
        blob = read_image_bytes(src)
        if not blob:
            log_event('download_fail', cam, src=src)
            return

        frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            log_event('decode_fail', cam, src=src)
            return

        H, W = frame.shape[:2]
        raw_frame = frame.copy()  # 原圖保存

        res = model(frame, conf=CONF_THRES, verbose=False)[0]
        masks = res.masks.data.cpu().numpy() if getattr(res, 'masks', None) is not None else None

        # ---- 本幀統計 ----
        area_now   = {k: 0 for k in LABELS}
        conf_now   = {k: 0.0 for k in LABELS}
        masks_now  = {k: None for k in LABELS}
        rocks_now  = []

        for i, box in enumerate(res.boxes):
            rawlbl = (getattr(res, 'names', None) or model.names)[int(box.cls[0])]
            lbl = CLASS_MAP.get(rawlbl)
            if lbl is None:  # 不在統一類別表
                continue

            conf = float(box.conf[0])
            conf_now[lbl] = max(conf_now[lbl], conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORMAP[lbl], 2)
            cv2.putText(frame, f'{lbl} {conf:.2f}', (x1, max(12, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORMAP[lbl], 2)

            if masks is not None and i < len(masks):
                m = cv2.resize(masks[i], (W, H), cv2.INTER_NEAREST).astype(bool)
                area_now[lbl] += int(m.sum())
                masks_now[lbl] = m if masks_now[lbl] is None else (masks_now[lbl] | m)
                # overlay = np.zeros_like(frame); overlay[m] = COLORMAP[lbl]
                # frame   = cv2.addWeighted(frame, 1, overlay, 0.35, 0)

            if lbl == 'large_rock':
                rocks_now.append([x1,y1,x2,y2])

        img_area = float(H * W) if H and W else 1.0
        prop_now = {k: (area_now[k] / img_area) for k in LABELS}

        # 混濁與清水是否重疊
        overlap_mc = False
        if masks_now['muddy_water'] is not None and masks_now['clear_water'] is not None:
            overlap_mc = bool((masks_now['muddy_water'] & masks_now['clear_water']).any())

        # ---- Rock 追蹤 ----
        rock_moved = False
        dq = prev_rock_boxes[cam]
        if dq and rocks_now:
            for pb in dq[-1]:
                if max(iou_box(pb, rb) for rb in rocks_now) < IOU_TH_ROCK:
                    rock_moved = True
                    break
        prev_move_cnt[cam] = prev_move_cnt[cam] + 1 if rock_moved else 0
        rock_moved = prev_move_cnt[cam] >= MOVE_CONSEC
        dq.append(rocks_now)
        if len(dq) > MOVE_CONSEC:
            dq.popleft()

        # ---- 燈號邏輯 ----
        now = time.time()
        cur, exp = alert_state[cam]

        def present(lbl):
            return (prop_now[lbl] >= MIN_PROP) or (conf_now[lbl] >= BASE_CONF)

        has_clear  = present('clear_water')
        has_muddy  = present('muddy_water')
        has_debris = present('debris_flow')

        prev_alert = cur
        red_reason = ''; yellow_reason = ''

        if cur == 'red' and now < exp:
            alert = 'red'
            red_reason = camera_status.get(cam, {}).get('redReason','')
            # 鎖定期間補回起始時間。exp 就是 start + HOLD_RED_SEC，所以這是精確
            # 反推而非估計；沒有它，重啟後回復的紅燈會永遠說不出事件何時發生。
            if not red_since_ts.get(cam):
                red_since_ts[cam] = exp - HOLD_RED_SEC
        else:
            base = 'green' if (has_clear or (not has_muddy and not has_debris)) else 'yellow'

            if has_muddy and has_clear:
                if overlap_mc:
                    base = 'green' if conf_now['clear_water'] >= conf_now['muddy_water'] else 'yellow'
                else:
                    base = 'green' if prop_now['clear_water'] >= prop_now['muddy_water'] else 'yellow'

            if cur == 'yellow':
                if has_muddy:
                    last_muddy_seen_ts[cam] = now
                if (not has_muddy and has_clear) or (not has_muddy and (now - last_muddy_seen_ts[cam] > YELLOW_CLEAR_SEC)):
                    base = 'green'

            # ★ 刻意的去彈跳（debounce）：紅燈只從「已經是黃燈」的狀態升上去。
            #   由綠燈直接偵測到 debris_flow 時，本輪先進黃燈，需連續兩輪才會轉紅。
            #   目的是避免單一雜訊幀就觸發長達一天（HOLD_RED_SEC）的紅燈鎖定。
            #   與 MOVE_CONSEC=3、MIN_YELLOW_SEC、YELLOW_CLEAR_SEC 是同一套遲滯設計。
            #   注意：離線版 debris_offline_inference.py 的 Judge 是「一偵測到就轉紅」，
            #   兩者判定不同屬已知且刻意的差異，勿逕自改成一致。
            go_red = (cur == 'yellow') and (has_debris or rock_moved)

            if go_red:
                alert_state[cam] = ('red', now + HOLD_RED_SEC)
                red_since_ts[cam] = now
                alert = 'red'
                red_reason = 'debris_flow' if has_debris else 'large_rock_move'
            else:
                if base == 'yellow' and cur != 'yellow':
                    yellow_since_ts[cam] = now
                if base == 'green' and cur == 'yellow' and (now - yellow_since_ts[cam] < MIN_YELLOW_SEC):
                    base = 'yellow'
                alert_state[cam] = (base, 0.0)
                if cur == 'red':
                    red_since_ts.pop(cam, None)   # 鎖定到期，本次紅燈事件結束
                alert = base
                yellow_reason = ('muddy_vs_clear_overlap' if (base=='yellow' and overlap_mc) else
                                 'muddy_dominant' if base=='yellow' else '')

        # ---- OSD ----
        y0 = 60
        for lbl in LABELS:
            txt = f'{lbl:<12} P={prop_now[lbl]:.4f}  C={conf_now[lbl]:.2f}'
            cv2.putText(frame, txt, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORMAP[lbl], 2)
            y0 += 22
        cv2.putText(frame, f'Alert:{alert}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    {'green':(0,255,0),'yellow':(0,255,255),'red':(0,0,255)}[alert], 3)

        # ---- 準備存檔 ----
        keep_under(SAVE_ROOT, MAX_STORAGE)
        outdir  = os.path.join(SAVE_ROOT, cam)
        os.makedirs(outdir, exist_ok=True)
        ts_tag  = time.strftime("%Y%m%d_%H%M%S")
        fname_det = f'{cam}_{ts_tag}_{alert}.jpg'        # 偵測圖
        fname_raw = f'{cam}_{ts_tag}_{alert}_raw.jpg'    # 原圖

        # 寫檔
        path_raw = os.path.join(outdir, fname_raw)
        path_det = os.path.join(outdir, fname_det)
        cv2.imwrite(path_raw, raw_frame)
        cv2.imwrite(path_det, frame)
        try:
            note_new_bytes(os.path.getsize(path_raw) + os.path.getsize(path_det))
        except OSError:
            pass

        # URL
        # ★ 一律存「相對路徑」而非絕對網址。原本寫入 http://127.0.0.1:8000/...，
        #   只有在本機且剛好用 8000 埠瀏覽時才顯示得出影像；換埠、從區網其他機器
        #   或手機開啟時，127.0.0.1 會指向觀看者自己的電腦而全部破圖。
        #   （雲端版之所以正常，是因為 sync_to_cloud.rewrite_urls() 上傳前又把前綴拿掉。）
        #   相對路徑會自動對應到當下提供頁面的來源，本機與雲端都正確。
        cam_q   = quote(cam);   det_q = quote(fname_det); raw_q = quote(fname_raw)
        det_url = f'/saved_images/{cam_q}/{det_q}'
        raw_url = f'/saved_images/{cam_q}/{raw_q}'

        # history（舊→新；受控於 HISTORY_KEEP_SEC / HISTORY_MAX）
        now_ts = time.time()

        qd = hist_queues_det[cam]; qd.append((now_ts, det_url))
        while qd and (now_ts - qd[0][0] > HISTORY_KEEP_SEC or len(qd) > HISTORY_MAX):
            qd.popleft()

        qr = hist_queues_raw[cam]; qr.append((now_ts, raw_url))
        while qr and (now_ts - qr[0][0] > HISTORY_KEEP_SEC or len(qr) > HISTORY_MAX):
            qr.popleft()

        ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prop_out = {k: round(float(prop_now[k]), 4) for k in LABELS}
        conf_out = {k: round(float(conf_now[k]), 3) for k in LABELS}

        # 紅燈的時間軸。timestamp 是「最後一次判讀」，每輪都會更新；沒有這兩個
        # 欄位的話，凌晨鎖定的紅燈和此刻正在發生的紅燈在網頁上長得一模一樣。
        _lvl, _exp = alert_state[cam]
        _since = red_since_ts.get(cam, 0.0)
        red_since_str  = (datetime.fromtimestamp(_since).strftime('%Y-%m-%d %H:%M:%S')
                          if alert == 'red' and _since else None)
        hold_until_str = (datetime.fromtimestamp(_exp).strftime('%Y-%m-%d %H:%M:%S')
                          if _lvl == 'red' and _exp else None)

        write_camera_status(cam, {
            'alert'     : alert,
            'timestamp' : ts_str,
            'redSince'    : red_since_str,
            'redHoldUntil': hold_until_str,
            'reportInterval': note_report(cam, time.time()),
            'imageUrl'  : det_url,
            'rawUrl'    : raw_url,
            'history'   : [u for _, u in qd],
            'historyRaw': [u for _, u in qr],
            'proportion': prop_out,
            'confidence': conf_out,
            'overlap_muddy_clear': bool(overlap_mc),
            'yellowReason': yellow_reason,
            'redReason'   : red_reason
        })

        # ---- logs ----
        append_metrics(cam, ts_str, prop_out, conf_out, alert, yellow_reason, red_reason)
        if LOG_EVERY_FRAME:
            try:
                log_frame_summary(cam, res)
            except Exception:
                pass

        if alert_state[cam][0] != prev_alert:
            log_event('alert_change', cam, prev=prev_alert, curr=alert, yellowReason=yellow_reason, redReason=red_reason)
            log.info(f'{cam} alert {prev_alert} -> {alert} ({red_reason or yellow_reason})')
            save_alert_state()   # 只在燈號改變時落地，一天約數百次而非每秒數十次
        else:
            log_event('frame', cam, alert=alert)

    except Exception as e:
        log.error(f'{cam} exception {e}', exc_info=True)
        log_event('exception', cam, message=str(e))

# ================== Worker（最早到期排程） ================== #
def worker_normal(cam_items: list):
    """
    以「最早到期」方式排程每台攝影機，下次偵測時間依當前燈號決定：
      紅 > 黃 > 綠（間隔最短 -> 偵測最頻繁）
    """
    driver = None
    try:
        # ★ 建立 driver 移入 try：原本在 try 之外，啟動失敗時例外會直接逸出本函式，
        #   既不會被下方 except 記錄，finally 也不會執行，執行緒就這樣無聲死亡。
        driver = create_driver(headless=args.headless)
        cam_url = {cam: url for cam, url in cam_items}
        now = time.time()
        next_due = {cam: now for cam, _ in cam_items}

        while True:
            if not next_due:
                time.sleep(0.5)
                continue

            cam = min(next_due, key=lambda k: next_due[k])
            due = next_due[cam]
            now = time.time()

            if now < due:
                time.sleep(min(0.2, due - now))
                continue

            try:
                detect_once(cam, cam_url[cam], driver)
            except Exception as e:
                log.error(f'worker detect_once error on {cam}: {e}', exc_info=True)

            interval = _desired_interval_for(cam)
            next_due[cam] = time.time() + interval

    except Exception as e:
        log.error(f'Worker fatal: {e}', exc_info=True)
    finally:
        if driver is not None:
            try: driver.quit()
            except Exception: pass

# ================== main ================== #
if __name__ == '__main__':
    load_alert_state()   # 讓 24 小時紅燈鎖定跨重啟存活

    with open(CCTV_JSON, encoding='utf-8') as f:
        cams = json.load(f)

    # 篩選 focus
    cam_list = []
    for c in cams:
        name, url = c['攝影機名稱'], c['影像連結網址']
        if FOCUS_LIST and name not in FOCUS_LIST:
            continue
        cam_list.append((name, url))

    if not cam_list:
        log.warning('No cameras to watch.')
        sys.exit(0)

    chunks = [[] for _ in range(min(THREADS, len(cam_list)))]
    for i, item in enumerate(cam_list):
        chunks[i % len(chunks)].append(item)

    threads = []
    for idx, chunk in enumerate(chunks):
        t = threading.Thread(target=worker_normal, args=(chunk,), daemon=True, name=f"worker-{idx+1}")
        t.start()
        threads.append(t)
        log.info(f'Worker-{idx+1} handling {len(chunk)} cameras')

    # ★ 監看 worker 存活：原本主迴圈只是永遠 sleep，即使所有 worker 都已死亡，
    #   程序仍會「看似正常」地活著，但 camera_status.json 永遠不再更新。
    try:
        alive_prev = len(threads)
        while True:
            time.sleep(5)
            alive = [t for t in threads if t.is_alive()]
            if len(alive) != alive_prev:
                dead = [t.name for t in threads if not t.is_alive()]
                log.error(f'Worker 異常結束：{dead}（存活 {len(alive)}/{len(threads)}）')
                alive_prev = len(alive)
            if not alive:
                log.critical('所有 worker 皆已結束，已無任何攝影機被監看，程序即將離開。')
                sys.exit(1)
    except KeyboardInterrupt:
        log.info('Stop by user')
