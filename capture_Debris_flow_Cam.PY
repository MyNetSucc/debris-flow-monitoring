#!/usr/bin/env python3
# yolo11m_seg_detectfromcctv_nooverlay.py
# -------------------------------------------------
#   • 執行 YOLO 推論、警戒邏輯、camera_status.json
#   • 不在影像上畫任何遮罩／文字（純原始 CCD 畫面）
# -------------------------------------------------
BASE_URL = 'http://127.0.0.1:8000'

import os, time, json, threading, queue, logging, argparse, urllib3
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import cv2, requests
from ultralytics import YOLO
from selenium import webdriver
from urllib.parse import quote
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by  import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support    import expected_conditions as EC
from requests.exceptions import ReadTimeout, ConnectionError, SSLError
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------- 旗標：是否在畫面上標註 --------
DRAW = False              # ← 只要關閉就不畫任何東西

# ============ CLI ============
ap = argparse.ArgumentParser(description='Debris-flow realtime detector (no overlay)')
ap.add_argument('--focus', nargs='*', default=[], help='只監看指定攝影機 (空=全部)')
args       = ap.parse_args()
FOCUS_LIST = args.focus

# ============ 參數（其餘同原程式） ============
HIST_SEC, HIST_MAX = 30, 40
IOU_TH_ROCK, CENTER_TH_ROCK, ROCK_MOVE_PROP = 0.30, 0.35, 0.50
MOVE_CONSEC, MISS_TOLERATE = 3, 2
ALPHA_SMOOTH, MIN_AREA = 0.2, 5_000
TH_DPROP_YEL, TH_DCONF_RED, TH_DCONF_YEL, BASE_CONF = 0.40, 0.50, 0.50, 0.25
HOLD_RED, HOLD_YEL = 86400, 86400

THREADS = 5
SAVE_ROOT = r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\saved_images"
MAX_STORAGE = 100 * 1024**3
MODEL_PATH = r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\yolo11m-seg.pt"
CHROMEDRIVER = r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\chromedriver-win64\chromedriver.exe"
CAM_STATUS_JS = 'camera_status.json'

COLORMAP = {'debris-flow':(0,0,255),'muddy':(255,0,0),
            'river':(0,255,255),'rock':(0,255,0)}
RENAME_LABEL = {'riverbond':'river'}

# ============ logging ==========
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/app.log', encoding='utf-8'),
              logging.StreamHandler()])
log = logging.getLogger(__name__)

# ============ YOLO ============
model = YOLO(MODEL_PATH)
model.conf, model.iou = 0.50, 0.50               # ← 正常推論設定

# ============ Selenium =========
def create_driver(headless=True):
    opt = Options()
    if headless:
        opt.add_argument('--headless'); opt.add_argument('--disable-gpu')
    opt.add_argument('--no-sandbox')
    opt.add_argument('--disable-dev-shm-usage')
    opt.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(service=Service(CHROMEDRIVER), options=opt)

# ============ Utilities（與原程式相同） ============
def download(url, retry=3, timeout=20):
    for _ in range(retry):
        try:
            r = requests.get(url, timeout=timeout, verify=False)
            if r.status_code == 200:
                return r
        except (ReadTimeout, ConnectionError, SSLError):
            pass
        time.sleep(1)
    return None

def keep_under(folder, limit):
    while True:
        total = sum(os.path.getsize(os.path.join(dp,f))
                    for dp,_,fs in os.walk(folder) for f in fs)
        if total <= limit:
            break
        files = [(os.path.join(dp,f), os.path.getmtime(os.path.join(dp,f)))
                 for dp,_,fs in os.walk(folder) for f in fs]
        files.sort(key=lambda x: x[1])
        os.remove(files[0][0])

def iou_box(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-6)

def mask_iou(m1,m2):
    inter = np.logical_and(m1,m2).sum()
    union = np.logical_or(m1,m2).sum()
    return inter/(union+1e-6)

# ========= 狀態快取（同原） =========
prev_comp   = defaultdict(dict)
prev_area   = defaultdict(lambda:{k:0 for k in COLORMAP})
prev_conf   = defaultdict(lambda:{k:0 for k in COLORMAP})
smooth_da   = defaultdict(lambda:{k:0.0 for k in COLORMAP})
smooth_dc   = defaultdict(lambda:{k:0.0 for k in COLORMAP})

prev_rock_boxes = defaultdict(deque)
prev_move_cnt   = defaultdict(int)
alert_state     = defaultdict(lambda:('green',0))

camera_status   = {}
hist_queues     = defaultdict(deque)

# ========= 主偵測流程（只把繪圖放 DRAW） =========
def detect_once(cam, url):
    if FOCUS_LIST and cam not in FOCUS_LIST:
        return
    driver = create_driver()
    try:
        driver.get(url)
        src = WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.ID, 'img1'))
        ).get_attribute('src')
        if 'noCCD_2.jpg' in src:
            return
        r = download(src)
        if not r: return
        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return

        # -------- YOLO 推論 --------
        H,W = frame.shape[:2]
        res  = model(frame)[0]
        masks = res.masks.data.cpu().numpy() if res.masks else None

        area_now = {k:0 for k in COLORMAP}
        conf_now = {k:0 for k in COLORMAP}
        comps_now={'river':[],'muddy':[]}
        rocks_now=[]
        debris_union=np.zeros((H,W),dtype=bool)

        for i,box in enumerate(res.boxes):
            raw  = model.names[int(box.cls[0])]
            lbl  = RENAME_LABEL.get(raw,raw)
            if lbl not in COLORMAP: continue
            conf = float(box.conf[0])
            conf_now[lbl] = max(conf_now[lbl], conf)

            if masks is not None:
                m = cv2.resize(masks[i], (W,H), cv2.INTER_NEAREST).astype(bool)
                area_now[lbl] += int(m.sum())
                if lbl in ('river','muddy'):
                    comps_now[lbl].append({'mask':m,'area':int(m.sum())})
                if lbl == 'debris-flow':
                    debris_union |= m
                if DRAW:
                    overlay=np.zeros_like(frame); overlay[m]=COLORMAP[lbl]
                    frame=cv2.addWeighted(frame,1,overlay,0.35,0)

            if lbl == 'rock':
                x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
                rocks_now.append({'bbox':[x1,y1,x2,y2]})
                if DRAW:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),COLORMAP[lbl],2)

            if DRAW:
                x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
                cv2.rectangle(frame,(x1,y1),(x2,y2),COLORMAP[lbl],2)
                cv2.putText(frame,f'{lbl} {conf:.2f}',
                            (x1,max(12,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.55,
                            COLORMAP[lbl],2)

        # （以下 ΔA / ΔC、警戒判定… 完全照舊，省略重貼）
        # ... 你的原始統計與判定邏輯 ...

        # ----------- 最後寫檔 -----------
        keep_under(SAVE_ROOT, MAX_STORAGE)
        fname = f'{cam}_{datetime.now():%Y%m%d_%H%M%S}.jpg'
        outdir = os.path.join(SAVE_ROOT, cam)
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, fname), frame)

    except Exception as e:
        log.error(f'{cam} exception {e}', exc_info=True)
    finally:
        driver.quit()

# ========= Thread worker =========
def worker(q: queue.Queue):
    while True:
        try:
            cam, url = q.get(timeout=10)
        except queue.Empty:
            break
        detect_once(cam, url)
        q.put((cam, url))
        q.task_done()
        time.sleep(1)

# ========= main =========
if __name__ == '__main__':
    with open('cctv.json', encoding='utf-8') as f:
        cams = json.load(f)

    q = queue.Queue()
    for c in cams:
        name, url = c['攝影機名稱'], c['影像連結網址']
        if FOCUS_LIST and name not in FOCUS_LIST:
            continue
        q.put((name, url))

    for _ in range(min(THREADS, q.qsize() or 1)):
        threading.Thread(target=worker, args=(q,), daemon=True).start()

    try:
        q.join()
    except KeyboardInterrupt:
        log.info('Stop by user')
