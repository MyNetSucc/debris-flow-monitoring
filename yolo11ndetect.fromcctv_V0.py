import os
import time
import json
import threading
import requests
import logging
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from requests.exceptions import ReadTimeout, ConnectionError

# -------------------- 配置日誌 --------------------
# 創建日誌文件夾
log_folder = os.path.join(os.getcwd(), "logs")
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "app.log")

# 配置 logging 模組
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- 加載模型 --------------------
# 請確保模型路徑正確
model_path = "C:\\Users\\MyNetSucc\\Downloads\\debris_flow_detect_V0.1\\yolo11n debris_flow_model.pt"
model = YOLO(model_path)
model.conf = 0.6  # 設置置信度閾值為 0.5
model.iou = 0.5   # 設置 NMS 閾值為 0.5

# -------------------- 配置 Selenium WebDriver --------------------
def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 無頭模式
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    # 請確保 ChromeDriver 路徑正確
    service = Service('C:\\Users\\MyNetSucc\\Downloads\\debris_flow_detect_V0.1\\chromedriver-win64\\chromedriver.exe')
    return webdriver.Chrome(service=service, options=chrome_options)

# -------------------- 創建輸出文件夾 --------------------
base_output_folder = "C:\\Users\\MyNetSucc\\Downloads\\debris_flow_detect_V0.1\\saved_images"
os.makedirs(base_output_folder, exist_ok=True)

# -------------------- 初始化檢測計數 --------------------
detection_counts = defaultdict(int)

# -------------------- 定義帶有重試機制的圖像下載函數 --------------------
def download_image_with_retry(url, retries=3, timeout=20):
    for attempt in range(retries):
        try:
            img_response = requests.get(url, timeout=timeout)
            if img_response.status_code == 200:
                return img_response
            else:
                logger.warning(f"HTTP 狀態碼: {img_response.status_code}")
        except (ReadTimeout, ConnectionError) as e:
            logger.warning(f"嘗試第 {attempt + 1}/{retries} 次：從 {url} 獲取圖像時發生錯誤：{e}")
            if attempt < retries - 1:
                time.sleep(5)  # 等待一段時間再重試
            else:
                raise
    return None

# -------------------- 定義函數來計算資料夾的總大小 --------------------
def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

# -------------------- 定義函數來刪除最舊的文件以控制總大小 --------------------
def manage_storage_limit(folder, max_size_bytes):
    while True:
        total_size = get_folder_size(folder)
        if total_size <= max_size_bytes:
            break
        # 獲取所有文件的路徑和修改時間
        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    files.append((fp, os.path.getmtime(fp)))
        if not files:
            break
        # 按修改時間排序，最舊的在前
        files.sort(key=lambda x: x[1])
        # 刪除最舊的文件
        oldest_file = files[0][0]
        try:
            os.remove(oldest_file)
            logger.info(f"刪除最舊的文件以控制儲存空間：{oldest_file}")
        except Exception as e:
            logger.error(f"刪除文件時發生錯誤：{e}")
            break

# -------------------- 處理單個攝像機的函數 --------------------
def process_cctv(cctv_name, cctv_url):
    logger.info(f"開始處理攝像機：{cctv_name}")
    driver = create_webdriver()
    driver.get(cctv_url)
    retry_count = 0
    max_retries = 5  # 設置最大重試次數
    wait_time = 10   # 等待時間（秒）
    max_storage_size = 5 * 1024 * 1024 * 1024  # 最大儲存空間 5GB
    try:
        while True:
            try:
                # 等待圖像元素加載
                img_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "img1"))
                )

                # 獲取圖像源
                img_src = img_element.get_attribute("src")

                # 檢查圖像是否為佔位符
                if "noCCD_2.jpg" in img_src:
                    logger.warning(f"{cctv_name}: 影像暫時不可用（佔位符圖像）。")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"{cctv_name}: 因為佔位符圖像，已達到最大重試次數。暫時跳過此攝像機。")
                        break  # 跳出循環，結束當前線程
                    time.sleep(wait_time)
                    continue

                # 使用帶有重試機制的函數下載圖像
                try:
                    img_response = download_image_with_retry(img_src, retries=3, timeout=20)
                except Exception as e:
                    logger.error(f"{cctv_name}: 獲取圖像時發生致命錯誤：{e}", exc_info=True)
                    break

                if img_response is None:
                    logger.error(f"{cctv_name}: 無法獲取圖像，已超過最大重試次數。")
                    break

                img_array = np.array(bytearray(img_response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # 檢查圖像是否有效
                if img is None:
                    logger.warning(f"{cctv_name}: 收到無效的圖像數據。")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"{cctv_name}: 因為無效的圖像數據，已達到最大重試次數。暫時跳過此攝像機。")
                        break
                    time.sleep(wait_time)
                    continue

                # 重置重試計數器
                retry_count = 0

                # 執行模型推理
                results = model(img, conf=0.5, iou=0.5)

                # 處理檢測結果
                detection_made = False  # 標記是否有檢測結果
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            label = model.names[cls]
                            if label in ["debris-flow", "rock", "muddy"]:
                                detection_counts[label] += 1
                                # 繪製邊界框和標籤
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text = f"{label}: {conf:.2f}"
                                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                logger.info(f"在 {cctv_name} 檢測到 {label}: 坐標 {x1}, {y1}, {x2}, {y2}.")
                                detection_made = True
                        if detection_made:
                            # 在保存圖像之前，管理儲存空間
                            manage_storage_limit(base_output_folder, max_storage_size)
                            # 保存帶有繪製結果的圖像
                            cctv_folder = os.path.join(base_output_folder, cctv_name)
                            os.makedirs(cctv_folder, exist_ok=True)
                            filename = f"{label}_{cctv_name}_{int(time.time())}.jpg"
                            filepath = os.path.join(cctv_folder, filename)
                            cv2.imwrite(filepath, img)
                            logger.info(f"圖像已保存到 {filepath}")
                            break  # 只處理一次檢測

                time.sleep(5)  # 等待下一次檢測

            except (ReadTimeout, ConnectionError):
                logger.warning(f"{cctv_name}: 讀取操作超時或連接錯誤。重試中...")
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"{cctv_name}: 因為讀取超時或連接錯誤，已達到最大重試次數。暫時跳過此攝像機。")
                    break
                time.sleep(wait_time)
                continue

            except TimeoutException:
                logger.warning(f"{cctv_name}: 等待圖像元素超時。重試中...")
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"{cctv_name}: 因為超時，已達到最大重試次數。暫時跳過此攝像機。")
                    break
                time.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(f"{cctv_name}: 預測時發生錯誤: {str(e)}", exc_info=True)
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"{cctv_name}: 因為錯誤，已達到最大重試次數。暫時跳過此攝像機。")
                    break
                time.sleep(wait_time)
                continue

    finally:
        driver.quit()
        logger.info(f"結束處理攝像機：{cctv_name}")

# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    # 讀取 CCTV 數據
    with open("cctv.json", "r", encoding="utf-8") as f:
        cctv_data = json.load(f)

    # 控制線程數量
    max_threads = 5  # 根據系統資源調整
    threads = []
    for cctv in cctv_data:
        cctv_name = cctv.get("攝影機名稱")
        cctv_url = cctv.get("影像連結網址")
        if cctv_name and cctv_url:
            t = threading.Thread(target=process_cctv, args=(cctv_name, cctv_url))
            threads.append(t)
            t.start()
            # 控制線程數量
            while threading.active_count() > max_threads:
                time.sleep(1)

    # 等待所有線程完成
    for t in threads:
        t.join()

    # 輸出檢測結果
    logger.info("\n檢測結果彙總:")
    for label, count in detection_counts.items():
        logger.info(f"{label}: {count}")
