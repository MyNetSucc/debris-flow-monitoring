#!/usr/bin/env python3
# yolo11m_seg_detectfromcctv_V3.py
# -------------------------------------------------
#   • 類別統一：clear_water / muddy_water / debris_flow / large_rock
#   • camera_status.json 輸出 proportion(比例0-1) / confidence(最大信心)
#   • 黃/紅燈邏輯：紅燈 hold 一天、黃燈退場、紅燈不被覆蓋
#   • Rock 以 IoU 追蹤連續位移判定移動
#   • 保留 history（偵測圖）與 historyRaw（原圖）
#   • 日誌：logs/app.log (旋轉)、logs/events.jsonl、logs/metrics/<cam>.csv
#   • 效能：Selenium Driver 於 worker 內重用、requests.Session() 共用
#   • 新增：--conf / --site / --no-frame-log；每幀摘要（時間＋地點＋has_masks＋類別）
# -------------------------------------------------

import os
import sys
import time
import json
import threading
import logging
import argparse
import urllib3
import csv
from logging.handlers import RotatingFileHandler
from collections import deque, defaultdict, Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout, ConnectionError, SSLError
from urllib3.util.retry import Retry
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by  import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support    import expected_conditions as EC
from urllib.parse import quote, urljoin
from selenium.common.exceptions import TimeoutException, NoSuchElementException

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================== 統一命令列參數 ================== #
parser = argparse.ArgumentParser(description='Debris-flow realtime detector with logging & tuned logic')

# 原有參數
parser.add_argument('--focus', nargs='*', default=[], help='只監看指定攝影機 (空=全部)')
parser.add_argument('--threads', type=int, default=5)
parser.add_argument('--interval', type=float, default=1.0, help='每台攝影機輪詢間隔(秒)')
parser.add_argument('--log-level', default='INFO', choices='DEBUG INFO WARNING ERROR CRITICAL'.split())
parser.add_argument('--headless', action='store_true', default=True)
parser.add_argument('--model', default=r"C:\Users\MyNetSucc\Downloads\debris_yolo11m_colab\weights\best.pt")
parser.add_argument('--save-root', default=r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\saved_images")
parser.add_argument('--max-storage', type=float, default=100 * 1024**3)  # 100 GB
parser.add_argument('--chromedriver', default=r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\chromedriver-win64\chromedriver.exe")
parser.add_argument('--status-json', default='camera_status.json')
# 新增參數
parser.add_argument('--conf', type=float, default=0.35, help='YOLO confidence threshold')
parser.add_argument('--site', type=str, default='unknown_site', help='Human-readable site/location')
parser.add_argument('--no-frame-log', action='store_true', help='Disable per-frame logging')

args = parser.parse_args()

# 參數展開
FOCUS_LIST      = args.focus
THREADS         = args.threads
SAVE_ROOT       = args.save_root
MAX_STORAGE     = args.max_storage
MODEL_PATH      = args.model
CHROMEDRIVER    = args.chromedriver
CAM_STATUS_JS   = args.status_json
INTERVAL_SEC    = max(0.2, float(args.interval))
CONF_THRES      = args.conf
SITE_NAME       = args.site
LOG_EVERY_FRAME = not args.no_frame_log

# 其他環境設定
BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:8000')

# ================== Logging（旋轉檔） ================== #
os.makedirs('logs', exist_ok=True)
app_handler = RotatingFileHandler('logs/app.log', maxBytes=10_000_000, backupCount=5, encoding='utf-8')
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s',
    handlers=[app_handler, logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('df-app')

EVENTS_JSONL = 'logs/events.jsonl'
METRICS_DIR  = 'logs/metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

# ================== 判定門檻與常數 ================== #
# Rock 追蹤與配對
IOU_TH_ROCK   = 0.30
MOVE_CONSEC   = 3
# 判定門檻（注意：與 YOLO 推論 --conf 不同概念）
BASE_CONF     = 0.539   # 基礎信心門檻（自訂邏輯用）
MIN_PROP      = 0.0008  # 最小比例門檻（0.08%）
# 黃/紅燈時間
HOLD_RED_SEC     = 86400
MIN_YELLOW_SEC   = 8
YELLOW_CLEAR_SEC = 90

# ================== 類別與外觀 ================== #
LABELS = ['clear_water', 'muddy_water', 'debris_flow', 'large_rock']
CLASS_MAP = {
    # 同名直通（模型輸出已是統一名時）
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
    'debris_flow': (0, 0, 255),    # BGR
    'muddy_water': (255, 0, 0),
    'clear_water': (0, 255, 255),
    'large_rock' : (0, 255, 0)
}

# ================== 事件與指標寫檔 ================== #
def log_event(event: str, cam: str, **payload):
    """寫入結構化事件到 JSONL"""
    rec = {'ts': datetime.utcnow().isoformat()+'Z', 'event': event, 'camera': cam}
    rec.update(payload)
    with open(EVENTS_JSONL, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def append_metrics(cam: str, timestamp: str, prop: dict, conf: dict, alert: str, y_reason: str, r_reason: str):
    """每幀寫入一行 CSV，方便 Excel / pandas 讀取"""
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
model.conf = CONF_THRES  # 與命令列一致
model.iou  = 0.50

# 共用 requests.Session()（多執行緒連線池）
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

def keep_under(folder, limit):
    """將資料夾容量維持在 limit 以下（最舊檔案優先刪除）"""
    while True:
        total = 0
        for dp, _, fs in os.walk(folder):
            for f in fs:
                total += os.path.getsize(os.path.join(dp, f))
        if total <= limit:
            break
        files = []
        for dp, _, fs in os.walk(folder):
            for f in fs:
                p = os.path.join(dp, f)
                files.append((p, os.path.getmtime(p)))
        if not files:
            break
        files.sort(key=lambda x: x[1])  # 最舊優先
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

# ================== 全域狀態 ================== #
hist_queues        = defaultdict(deque)              # cam -> deque[(ts,url)] 偵測圖
hist_queues_raw    = defaultdict(deque)              # cam -> deque[(ts,url)] 原圖
prev_rock_boxes    = defaultdict(deque)              # cam -> deque[[x1,y1,x2,y2],...]*N
prev_move_cnt      = defaultdict(int)                # cam -> 連續移動幀數
alert_state        = defaultdict(lambda: ('green', 0.0))  # cam -> (level, expire_ts)
yellow_since_ts    = defaultdict(float)              # cam -> 黃燈起始時間
last_muddy_seen_ts = defaultdict(float)              # cam -> 最近看到 muddy 時間
camera_status      = {}                              # 最新 camera_status.json 快取
_no_mask_warned    = set()                           # 每台攝影機僅提醒一次

# ================== 逐幀摘要（時間＋地點＋類別＋has_masks） ================== #
def log_frame_summary(cam_name, res, frame_idx=None, frame_ms=None):
    """
    cam_name: 攝影機名稱/來源字串
    res:      Ultralytics 單幀推論結果 (model(...)[0])
    frame_idx:整數，當前幀號；沒有可傳 None
    frame_ms: 視訊內毫秒位置；即時串流可 None
    """
    names_src = getattr(res, 'names', None) or model.names
    labels = []
    for b in getattr(res, 'boxes', []):
        raw = names_src[int(b.cls[0])]
        lbl = CLASS_MAP.get(raw, raw)  # 找不到就用原名
        labels.append(lbl)

    has_masks = getattr(res, 'masks', None) is not None
    counts = dict(Counter(labels))

    # 視訊相對時間字串（若有）
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

# ================== Selenium ================== #
def create_driver(headless=True):
    opt = Options()
    if headless:
        opt.add_argument('--headless=new')
        opt.add_argument('--disable-gpu')
    opt.add_argument('--no-sandbox')
    opt.add_argument('--disable-dev-shm-usage')
    opt.add_argument('--window-size=1920,1080')
    # 加速：不載入圖片（我們只拿 src）
    prefs = {"profile.managed_default_content_settings.images": 2}
    opt.add_experimental_option("prefs", prefs)
    opt.page_load_strategy = 'eager'
    return webdriver.Chrome(service=Service(CHROMEDRIVER), options=opt)

def _scan_for_media_src(driver):
    """在當前 browsing context 掃描可能的影像來源並回傳第一個非空 src。"""
    # 優先常見 id
    for css in ["img#img1", "img#Image1", "img#camimg"]:
        els = driver.find_elements(By.CSS_SELECTOR, css)
        for e in els:
            src = (e.get_attribute('src') or '').strip()
            if src:
                return src

    # 任何 <img>
    for e in driver.find_elements(By.TAG_NAME, 'img'):
        src = (e.get_attribute('src') or '').strip()
        if src:
            return src

    # <video><source> or <video src>
    for v in driver.find_elements(By.TAG_NAME, 'video'):
        src = (v.get_attribute('src') or '').strip()
        if src:
            return src
        for s in v.find_elements(By.TAG_NAME, 'source'):
            ss = (s.get_attribute('src') or '').strip()
            if ss:
                return ss

    # <canvas data-url / data-src>
    for c in driver.find_elements(By.CSS_SELECTOR, 'canvas[data-url], canvas[data-src]'):
        src = (c.get_attribute('data-url') or c.get_attribute('data-src') or '').strip()
        if src:
            return src

    return None


def resolve_image_src(driver, page_url: str, wait_sec: float = 25.0) -> str | None:
    """
    嘗試在頁面與所有 iframe/frames 中找出一個可下載的影像來源 URL。
    回傳絕對網址，或 None。
    """
    t_end = time.time() + wait_sec
    last_err = None

    while time.time() < t_end:
        try:
            # 先掃描主頁面
            driver.switch_to.default_content()
            src = _scan_for_media_src(driver)
            if src:
                # 轉成絕對路徑
                abs_src = urljoin(page_url, src)
                return abs_src

            # 掃描 iframe / frame
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
                    # 繼續掃下一個 frame
                    driver.switch_to.default_content()

        except Exception as e:
            last_err = e

        time.sleep(0.5)

    # 逾時
    if last_err:
        log.debug(f"resolve_image_src last error: {last_err}")
    return None
# ================== 主偵測 ================== #
def detect_once(cam, url, driver: webdriver.Chrome):
    try:
        driver.get(url)

        # 嘗試在頁面/iframe 找到任何可用影像來源（不等特定 id）
        src = resolve_image_src(driver, url, wait_sec=25)

        if not src:
            log_event('no_image_element', cam, page=url)
            log.warning(f"{cam} 找不到可用影像元素或 src（頁面/iframe 皆無）")
            return

        if 'noCCD_2.jpg' in src:
            log_event('no_image', cam, page=url, src=src)
            return

        r = download(src)
        if r is None:
            log_event('download_fail', cam, src=src)
            return

        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            log_event('decode_fail', cam, src=src)
            return

        raw_frame = frame.copy()  # 保存原圖

        H, W = frame.shape[:2]
        res = model(frame, conf=CONF_THRES, verbose=False)[0]  # 以命令列 conf 執行

        # seg 權重提醒（每台攝影機只提醒一次）
        if getattr(res, 'masks', None) is None and cam not in _no_mask_warned:
            log.warning(f"[{SITE_NAME}] {cam}: 模型沒有輸出 segmentation masks；請確認載入的是 *seg* 權重。")
            _no_mask_warned.add(cam)

        # 逐幀摘要
        if LOG_EVERY_FRAME:
            log_frame_summary(cam_name=cam, res=res, frame_idx=None, frame_ms=None)

        masks = res.masks.data.cpu().numpy() if getattr(res, 'masks', None) is not None else None

        # ---- 本幀統計 ----
        area_now   = {k: 0 for k in LABELS}
        conf_now   = {k: 0.0 for k in LABELS}
        masks_now  = {k: None for k in LABELS}
        rocks_now  = []

        for i, box in enumerate(res.boxes):
            raw = (getattr(res, 'names', None) or model.names)[int(box.cls[0])]
            lbl = CLASS_MAP.get(raw)
            if lbl is None:
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
        now_ts = time.time()
        cur, exp = alert_state[cam]

        def present(lbl):
            return (prop_now[lbl] >= MIN_PROP) or (conf_now[lbl] >= BASE_CONF)

        has_clear  = present('clear_water')
        has_muddy  = present('muddy_water')
        has_debris = present('debris_flow')

        prev_alert = cur
        red_reason = ''
        yellow_reason = ''

        if cur == 'red' and now_ts < exp:
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
                    last_muddy_seen_ts[cam] = now_ts
                if (not has_muddy and has_clear) or (not has_muddy and (now_ts - last_muddy_seen_ts[cam] > YELLOW_CLEAR_SEC)):
                    base = 'green'

            go_red = (cur == 'yellow') and (has_debris or rock_moved)

            if go_red:
                alert_state[cam] = ('red', now_ts + HOLD_RED_SEC)
                alert = 'red'
                red_reason = 'debris_flow' if has_debris else 'large_rock_move'
            else:
                if base == 'yellow' and cur != 'yellow':
                    yellow_since_ts[cam] = now_ts
                if base == 'green' and cur == 'yellow' and (now_ts - yellow_since_ts[cam] < MIN_YELLOW_SEC):
                    base = 'yellow'
                alert_state[cam] = (base, 0.0)
                alert = base
                yellow_reason = ('muddy_vs_clear_overlap' if (base=='yellow' and overlap_mc) else
                                 'muddy_dominant' if base=='yellow' else '')

        # ---- Save & status ----
        keep_under(SAVE_ROOT, MAX_STORAGE)
        ts_name = time.strftime("%Y%m%d_%H%M%S")
        outdir  = os.path.join(SAVE_ROOT, cam)
        os.makedirs(outdir, exist_ok=True)

        # 同步輸出 原圖(raw) 與 偵測圖(det)
        fname_raw = f'{cam}_{ts_name}_{alert}_raw.jpg'
        fname_det = f'{cam}_{ts_name}_{alert}.jpg'  # 偵測圖沿用舊命名
        cv2.imwrite(os.path.join(outdir, fname_raw), raw_frame)
        cv2.imwrite(os.path.join(outdir, fname_det), frame)

        cam_q   = quote(cam);   fraw_q = quote(fname_raw);  fdet_q = quote(fname_det)
        raw_rel = f'/saved_images/{cam_q}/{fraw_q}'
        det_rel = f'/saved_images/{cam_q}/{fdet_q}'
        raw_url = f'{BASE_URL}{raw_rel}'
        det_url = f'{BASE_URL}{det_rel}'

        # 各自維護 history 佇列
        now_stamp = time.time()
        qh_det = hist_queues[cam]
        qh_raw = hist_queues_raw[cam]
        qh_det.append((now_stamp, det_url))
        qh_raw.append((now_stamp, raw_url))
        while qh_det and (now_stamp - qh_det[0][0] > 30 or len(qh_det) > 40): qh_det.popleft()
        while qh_raw and (now_stamp - qh_raw[0][0] > 30 or len(qh_raw) > 40): qh_raw.popleft()

        ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prop_out = {k: round(float(prop_now[k]), 4) for k in LABELS}
        conf_out = {k: round(float(conf_now[k]), 3) for k in LABELS}

        camera_status[cam] = {
            'alert'     : alert,
            'timestamp' : ts_str,
            'imageUrl'  : det_url,                          # 偵測圖
            'rawUrl'    : raw_url,                          # 原圖
            'history'   : [u for _, u in qh_det],           # 偵測圖歷史（相容）
            'historyRaw': [u for _, u in qh_raw],           # 原圖歷史
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
        if prev_alert != alert:
            log_event('alert_change', cam, prev=prev_alert, curr=alert, yellowReason=yellow_reason, redReason=red_reason)
            log.info(f'{cam} alert {prev_alert} -> {alert} ({red_reason or yellow_reason})')
        else:
            log_event('frame', cam, alert=alert)

    except Exception as e:
        log.error(f'{cam} exception {e}', exc_info=True)
        log_event('exception', cam, message=str(e))

# ================== Worker ================== #
def worker_normal(cam_items: list):
    driver = create_driver(headless=args.headless)
    try:
        while True:
            for cam, url in cam_items:
                detect_once(cam, url, driver)
                time.sleep(INTERVAL_SEC)
    except Exception as e:
        log.error(f'Worker fatal: {e}', exc_info=True)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# ================== main ================== #
if __name__ == '__main__':
    try:
        with open('cctv.json', encoding='utf-8') as f:
            cams = json.load(f)
    except Exception as e:
        log.error(f'Load cctv.json failed: {e}')
        sys.exit(1)

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

    # 平均分配到各 worker
    chunks = [[] for _ in range(min(THREADS, len(cam_list)))]
    for i, item in enumerate(cam_list):
        chunks[i % len(chunks)].append(item)

    # 啟動 workers
    for idx, chunk in enumerate(chunks):
        t = threading.Thread(target=worker_normal, args=(chunk,), daemon=True, name=f"worker-{idx+1}")
        t.start()
        log.info(f'Worker-{idx+1} handling {len(chunk)} cameras')

    # 主執行緒保持
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        log.info('Stop by user')
