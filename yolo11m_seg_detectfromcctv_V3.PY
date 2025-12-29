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
ap.add_argument('--model', default=resolve_path('weights/best.pt', r"C:\Users\eric0\Downloads\debris_yolo11m_colab\weights\best.pt"),
                help='YOLO 模型權重檔路徑 (預設: weights/best.pt)')
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
                help='前端可讀的根 URL (預設: http://127.0.0.1:8000)')

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

def keep_under(folder, limit):
    while True:
        total = 0
        for dp, _, fs in os.walk(folder):
            for f in fs:
                total += os.path.getsize(os.path.join(dp, f))
        if total <= limit: break
        files = []
        for dp, _, fs in os.walk(folder):
            for f in fs:
                p = os.path.join(dp, f)
                files.append((p, os.path.getmtime(p)))
        if not files: break
        files.sort(key=lambda x: x[1])
        try:
            os.remove(files[0][0])
            log.warning(f"Storage limit trimming: removed {files[0][0]}")
        except Exception as e:
            log.error(f"Remove file failed: {e}")
            break

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
yellow_since_ts      = defaultdict(float)     # cam -> 黃燈起始時間
last_muddy_seen_ts   = defaultdict(float)     # cam -> 最近看到 muddy 時間
camera_status        = {}                     # 最新 camera_status.json 快取

# ================== Selenium ================== #
def create_driver(headless=True):
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

    # ★ 改為顯式建立 driver 並設定 page load timeout
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER), options=opt)
    driver.set_page_load_timeout(25)
    return driver

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
                overlay = np.zeros_like(frame); overlay[m] = COLORMAP[lbl]
                frame   = cv2.addWeighted(frame, 1, overlay, 0.35, 0)

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

            go_red = (cur == 'yellow') and (has_debris or rock_moved)

            if go_red:
                alert_state[cam] = ('red', now + HOLD_RED_SEC)
                alert = 'red'
                red_reason = 'debris_flow' if has_debris else 'large_rock_move'
            else:
                if base == 'yellow' and cur != 'yellow':
                    yellow_since_ts[cam] = now
                if base == 'green' and cur == 'yellow' and (now - yellow_since_ts[cam] < MIN_YELLOW_SEC):
                    base = 'yellow'
                alert_state[cam] = (base, 0.0)
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
        cv2.imwrite(os.path.join(outdir, fname_raw), raw_frame)
        cv2.imwrite(os.path.join(outdir, fname_det), frame)

        # URL
        cam_q   = quote(cam);   det_q = quote(fname_det); raw_q = quote(fname_raw)
        det_rel = f'/saved_images/{cam_q}/{det_q}'
        raw_rel = f'/saved_images/{cam_q}/{raw_q}'
        det_url = f'{BASE_URL}{det_rel}'
        raw_url = f'{BASE_URL}{raw_rel}'

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

        camera_status[cam] = {
            'alert'     : alert,
            'timestamp' : ts_str,
            'imageUrl'  : det_url,
            'rawUrl'    : raw_url,
            'history'   : [u for _, u in qd],
            'historyRaw': [u for _, u in qr],
            'proportion': prop_out,
            'confidence': conf_out,
            'overlap_muddy_clear': bool(overlap_mc),
            'yellowReason': yellow_reason,
            'redReason'   : red_reason
        }
        with open(CAM_STATUS_JS, 'w', encoding='utf-8') as f:
            json.dump(camera_status, f, ensure_ascii=False, indent=2)

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
    driver = create_driver(headless=args.headless)
    try:
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
        try: driver.quit()
        except: pass

# ================== main ================== #
if __name__ == '__main__':
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

    for idx, chunk in enumerate(chunks):
        t = threading.Thread(target=worker_normal, args=(chunk,), daemon=True, name=f"worker-{idx+1}")
        t.start()
        log.info(f'Worker-{idx+1} handling {len(chunk)} cameras')

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        log.info('Stop by user')
