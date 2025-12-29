# focus_server.py ── Flask + Selenium 專門跑「即時檢視」的攝影機
import os, json, threading, time, cv2, numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by  import By
from selenium.webdriver.support    import expected_conditions as EC
from yolo11m_seg_detectfromcctv_V3 import (
    create_driver, download, process_frame, SAVE_ROOT,
    CAM_STATUS_JS, log, COLORMAP
)

# ---------- 讀 cctv.json ----------
with open('cctv.json', encoding='utf-8') as f:
    CCTV = {c['攝影機名稱']: c for c in json.load(f)}

# ---------- util ----------
def write_cam_status(cam, alert, img_url, meta, reason=''):
    """
    將即時辨識結果寫回 camera_status.json
    （與主程式共用同一份檔案，前端直接撈）
    """
    if not os.path.exists(CAM_STATUS_JS):
        data = {}
    else:
        with open(CAM_STATUS_JS, encoding='utf-8') as f:
            data = json.load(f)

    prev_img, prev_time = '', ''
    if cam in data:
        prev_img  = data[cam].get('imageUrl', '')
        prev_time = data[cam].get('timestamp', '')

    data[cam] = {
        'alert'        : alert,
        'imageUrl'     : img_url,
        'timestamp'    : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prevImage'    : prev_img,
        'prevTime'     : prev_time,
        'redReason'    : reason if alert == 'red'    else '',
        'yellowReason' : reason if alert == 'yellow' else '',
        # --------- 以下為即時視窗需要用到的欄位 ----------
        'delta_conf'        : meta.get('delta_conf' , {}),
        'delta_area_prop'   : meta.get('delta_area_prop', {}),
        'area'              : meta.get('area' , {}),
    }
    with open(CAM_STATUS_JS, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- Focus Worker ----------
def focus_loop(cam):
    url    = CCTV[cam]['影像連結網址']
    driver = create_driver(headless=True)
    log.info(f"[Focus] Start {cam}")
    try:
        while cam in focus_set:
            try:
                driver.get(url)
                img_el = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "img1")))
                src = img_el.get_attribute("src")
                if 'noCCD_2.jpg' in src:
                    time.sleep(1); continue
                resp = download(src, timeout=10)
                if resp is None:
                    time.sleep(1); continue
                arr   = np.frombuffer(resp.content, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    time.sleep(1); continue

                # ★ 執行即時推論（正確參數順序：frame 在前）
                annotated, alert, meta = process_frame(frame, cam)

                # ---------- 儲存圖片 ----------
                out_dir = os.path.join(SAVE_ROOT, cam)
                os.makedirs(out_dir, exist_ok=True)
                fname   = f"{cam}_focus.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), annotated)
                img_url = f"http://127.0.0.1:8000/saved_images/{cam}/{fname}"

                # ---------- 寫回 camera_status ----------
                reason = meta.get('reason', '')
                write_cam_status(cam, alert, img_url, meta, reason)

            except Exception as e:
                log.warning(f"[Focus] {cam} error: {e}")
            time.sleep(1)      # 影像更新週期
    finally:
        driver.quit()
        log.info(f"[Focus] Stop  {cam}")

# ---------- Flask ----------
app       = Flask(__name__)
focus_set = set()          # 目前被要求即時的攝影機
threads   = {}             # {cam: Thread}

@app.route('/focus/add', methods=['POST'])
def add_focus():
    cam = request.json.get('camera')
    if cam not in CCTV:
        return jsonify({'error':'camera not found'}), 404
    if cam not in focus_set:
        focus_set.add(cam)
        t = threading.Thread(target=focus_loop, args=(cam,), daemon=True)
        t.start()
        threads[cam] = t
    return jsonify({'status':'ok'})

@app.route('/focus/remove', methods=['POST'])
def remove_focus():
    cam = request.json.get('camera')
    focus_set.discard(cam)          # 讓 worker 自行跳出
    return jsonify({'status':'ok'})

# ---------- run ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
