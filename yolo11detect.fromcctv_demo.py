import os
import time
import json
import logging
import requests
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict

from ultralytics import YOLO

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)
from requests.exceptions import ReadTimeout, ConnectionError

# -------------------- 日誌設定 --------------------
log_folder = os.path.join(os.getcwd(), "logs")
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -------------------- 加載 YOLO 模型 --------------------
model_path = r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\yolo11mseg.pt"
model = YOLO(model_path)
model.conf = 0.5  # 信心閾值
model.iou = 0.5   # NMS 閾值

# -------------------- 設定輸出資料夾 --------------------
base_output_folder = r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\saved_images"
os.makedirs(base_output_folder, exist_ok=True)

# -------------------- 初始化偵測計數、攝影機狀態 --------------------
detection_counts = defaultdict(int)
camera_status_dict = {}
camera_status_path = "camera_status.json"

# -------------------- 計算資料夾的大小 --------------------
def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

# -------------------- 控制總儲存大小 --------------------
def manage_storage_limit(folder, max_size_bytes):
    while True:
        total_size = get_folder_size(folder)
        if total_size <= max_size_bytes:
            break
        # 找出所有檔案並按修改時間排序
        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    files.append((fp, os.path.getmtime(fp)))
        if not files:
            break
        files.sort(key=lambda x: x[1])  # 最舊的在前
        oldest_file = files[0][0]
        try:
            os.remove(oldest_file)
            logger.info(f"刪除最舊檔案：{oldest_file}")
        except Exception as e:
            logger.error(f"刪除檔案時發生錯誤：{e}")
            break

# -------------------- 下載圖像 (requests) --------------------
def download_image_with_retry(url, retries=3, timeout=20):
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.content  # 回傳 bytes
            else:
                logger.warning(f"HTTP 狀態碼: {resp.status_code}")
        except (ReadTimeout, ConnectionError) as e:
            logger.warning(f"第 {attempt+1}/{retries} 次下載失敗 {url}，錯誤：{e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise
    return None

# -------------------- 單支攝影機處理函式 --------------------
def process_cctv(driver, cctv_name, cctv_url, prev_rock_positions):
    """
    1) driver.get(cctv_url)
    2) 等待 <img id="img1">
    3) 取得其 src => 若是 noCCD_2.jpg => 跳過
    4) 下載該圖 => 用 YOLO 偵測 => 繪製 => 更新 camera_status.json
    5) 返回這次畫面中 rock 的位置，以供下次比較
    """
    logger.info(f"開始處理攝影機：{cctv_name}")

    retry_count = 0
    max_retries = 3
    wait_time = 5
    max_storage_size = 5 * 1024 * 1024 * 1024  # 5 GB

    while True:
        try:
            driver.get(cctv_url)  # 載入頁面
            # 等待 <img id="img1">
            img_element = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "img1"))
            )
            img_src = img_element.get_attribute("src")
            if not img_src:
                logger.warning(f"{cctv_name}: <img id='img1'> 沒有 src")
                return prev_rock_positions  # 直接跳出

            if "noCCD_2.jpg" in img_src:
                logger.warning(f"{cctv_name}: 佔位符 noCCD_2.jpg => 無影像")
                return prev_rock_positions

            # 下載該圖片
            img_bytes = download_image_with_retry(img_src, retries=3, timeout=20)
            if not img_bytes:
                logger.warning(f"{cctv_name}: 圖片下載失敗 => 跳過")
                return prev_rock_positions

            # 轉成 OpenCV 圖片
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                logger.warning(f"{cctv_name}: 影像內容無效 => 跳過")
                return prev_rock_positions

            # ---------- YOLO 偵測 ----------
            results = model(frame, conf=0.5, iou=0.5)

            curr_rocks = []
            alert_level = "green"
            debris_flow_detected = False
            muddy_detected = False
            detected_labels = set()
            detection_made = False
            max_conf_value = 0.0

            for result in results:
                boxes = result.boxes
                masks = result.masks
                if not boxes or len(boxes) == 0:
                    continue
                mask_data = masks.data.cpu().numpy() if masks else None

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model.names[cls]

                    if conf > max_conf_value:
                        max_conf_value = conf

                    if label in ["debris-flow", "rock", "muddy"]:
                        detection_counts[label] += 1
                        detected_labels.add(label)
                        # 繪製框
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        text = f"{label}:{conf:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0,255,0), 2)
                        detection_made = True

                        # 若有分割遮罩
                        if mask_data is not None:
                            m = mask_data[idx]
                            h, w = frame.shape[:2]
                            m = cv2.resize(m, (w,h), interpolation=cv2.INTER_NEAREST)
                            m = m.astype(bool)
                            color_map = {
                                "debris-flow": [0,0,255],
                                "rock": [0,255,0],
                                "muddy": [255,0,0]
                            }
                            c = np.array(color_map.get(label, [0,255,0]), dtype=np.uint8)
                            colored_mask = np.zeros_like(frame, dtype=np.uint8)
                            colored_mask[m] = c
                            frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

                        # rock => 記錄位置
                        if label == "rock":
                            center = ((x1 + x2)//2, (y1 + y2)//2)
                            curr_rocks.append({
                                "bbox":[x1,y1,x2,y2],
                                "center": center,
                                "ts": time.time()
                            })
                        elif label == "debris-flow":
                            debris_flow_detected = True
                        elif label == "muddy":
                            muddy_detected = True

            # rock 是否位移
            rock_moved = False
            for c_rock in curr_rocks:
                for p_rock in prev_rock_positions:
                    dx = c_rock["center"][0] - p_rock["center"][0]
                    dy = c_rock["center"][1] - p_rock["center"][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    w = c_rock["bbox"][2] - c_rock["bbox"][0]
                    h = c_rock["bbox"][3] - c_rock["bbox"][1]
                    bbox_size = np.sqrt(w*w + h*h)
                    if dist > bbox_size * 0.5:
                        rock_moved = True
                        logger.info(f"{cctv_name}: rock位移 => {c_rock['bbox']}")

            new_rock_positions = curr_rocks

            # 警戒綜合判斷
            if debris_flow_detected:
                if rock_moved or muddy_detected:
                    alert_level = "red"
                else:
                    alert_level = "yellow"
            elif rock_moved and muddy_detected:
                alert_level = "red"
            elif rock_moved or muddy_detected:
                alert_level = "yellow"
            else:
                alert_level = "green"

            if alert_level == "red":
                logger.warning(f"{cctv_name}: 紅色警戒!")
            elif alert_level == "yellow":
                logger.warning(f"{cctv_name}: 黃色警戒!")
            else:
                logger.info(f"{cctv_name}: 綠色燈號")

            cv2.putText(frame, f"Alert:{alert_level}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,255,0) if alert_level=='green' else (0,0,255),
                        2)

            # 控制儲存空間
            manage_storage_limit(base_output_folder, max_storage_size)

            # 儲存
            now_str = time.strftime("%Y%m%d_%H%M%S")
            if detection_made:
                labels_str = "_".join(detected_labels)
                filename = f"{cctv_name}_{now_str}_{labels_str}_{alert_level}.jpg"
            else:
                filename = f"{cctv_name}_{now_str}_{alert_level}.jpg"

            filename = filename.replace(":", "_").replace(" ", "_")
            cctv_folder = os.path.join(base_output_folder, cctv_name)
            os.makedirs(cctv_folder, exist_ok=True)
            filepath = os.path.join(cctv_folder, filename)
            cv2.imwrite(filepath, frame)
            logger.info(f"圖像已保存到: {filepath}")

            # YOLO完成時間
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 組圖片URL(若您有其他檔案伺服器，請自行對應)
            image_url = f"http://127.0.0.1:8000/saved_images/{cctv_name}/{filename}"

            # 寫入 camera_status.json
            camera_status_dict[cctv_name] = {
                "alert": alert_level,
                "imageUrl": image_url,
                "timestamp": detection_time,
                "confidence": round(max_conf_value,3)
            }
            with open(camera_status_path, "w", encoding="utf-8") as f:
                json.dump(camera_status_dict, f, ensure_ascii=False, indent=2)

            return new_rock_positions  # 本輪成功 => 返回

        except (ReadTimeout, ConnectionError):
            logger.warning(f"{cctv_name}: 連線超時/錯誤 => 重試中...")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"{cctv_name}: 超出重試 => 跳過")
                return prev_rock_positions
            time.sleep(wait_time)

        except TimeoutException:
            logger.warning(f"{cctv_name}: Selenium 載入超時 => 重試中...")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"{cctv_name}: 過多超時 => 跳過")
                return prev_rock_positions
            time.sleep(wait_time)

        except WebDriverException as e:
            logger.error(f"{cctv_name}: WebDriver 異常: {e.msg} => 重試中...")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"{cctv_name}: WebDriver失敗過多 => 跳過")
                return prev_rock_positions
            time.sleep(wait_time)

        except Exception as e:
            logger.error(f"{cctv_name}: 其它錯誤: {e}", exc_info=True)
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"{cctv_name}: 錯誤過多 => 跳過")
                return prev_rock_positions
            time.sleep(wait_time)

# -------------------- 主程式：單執行緒 (排隊式) --------------------
if __name__=="__main__":
    # 讀取攝影機清單
    with open("cctv.json","r",encoding="utf-8") as f:
        cctv_data= json.load(f)

    # 建立 Selenium driver (只此一個)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(r"C:\Users\MyNetSucc\Downloads\debris_flow_detect_V0.1\chromedriver-win64\chromedriver.exe")

    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)  # 加長頁面載入超時

    # Rock 的位置信息：dict => key=攝影機名稱, val=上一輪畫面的 rockList
    prev_rocks_map = {}

    try:
        # 單次: 逐個攝影機排隊處理
        for cctv in cctv_data:
            cctv_name = cctv.get("攝影機名稱")
            cctv_url  = cctv.get("影像連結網址")

            if not (cctv_name and cctv_url):
                continue

            old_rocks = prev_rocks_map.get(cctv_name, [])
            new_rocks = process_cctv(driver, cctv_name, cctv_url, old_rocks)
            prev_rocks_map[cctv_name] = new_rocks

            # 視需求，可在兩個攝影機之間 sleep 幾秒
            time.sleep(5)

        logger.info("\n=== 偵測結束 ===")
        for label, cnt in detection_counts.items():
            logger.info(f"{label}: {cnt}")

    finally:
        driver.quit()
        logger.info("程式結束。")
