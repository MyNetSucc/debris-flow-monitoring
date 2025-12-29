import os
os.system('pip install roboflow')
os.system('pip install selenium')
os.system('pip install tensorflow')
os.system('pip install onnxruntime')

import json
import cv2
import numpy as np
import roboflow as Roboflow
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import threading
import inference  # Import Roboflow's self-hosted inference library
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 設定 Roboflow 自我託管模型的 API 金鑰與模型
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("API key is not defined. Please check your .env file.")
model = inference.get_model("debris-flow-alvip/1", api_key=api_key)  # 替換為你的模型名稱

# 設定 Selenium WebDriver
def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 無頭模式
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--remote-debugging-port=9222")
    service = Service('C:\\Users\\eric0\\Downloads\\debris_flow_detect_V0.1\\chromedriver-win64\\chromedriver.exe')
    return webdriver.Chrome(service=service, options=chrome_options)


# 設定儲存影像的資料夾
base_output_folder = "C:\\Users\\eric0\\Downloads\\debris_flow_detect_V0.1\\saved_images"
os.makedirs(base_output_folder, exist_ok=True)

# 設定儲存空間的限制 (例如 1GB)
MAX_STORAGE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB

# 檢查儲存空間大小
def check_storage_space(output_folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# 處理 CCTV 影像的函式
def process_cctv(cctv_name, cctv_url):
    driver = create_webdriver()
    driver.get(cctv_url)

    try:
        # 等待影像載入
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "img"))
        )
        
        # 擷取影像並進行推論
        img_element = driver.find_element(By.TAG_NAME, "img")
        img_data = img_element.screenshot_as_png
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 使用模型進行推論
        results = model.predict(image=img, confidence=50, overlap=50)
        predictions = results.get("predictions", [])

        # 檢查是否有偵測到 debris-flow, rock 或 muddy
        for prediction in predictions:
            label = prediction.get("class")
            if label in ["debris-flow", "rock", "muddy"]:
                # 檢查儲存空間是否超過限制
                if check_storage_space(base_output_folder) >= MAX_STORAGE_SIZE:
                    print("儲存空間已滿，無法儲存更多影像。")
                    return
                
                # 儲存影像至對應的 CCTV 資料夾
                cctv_folder = os.path.join(base_output_folder, cctv_name)
                os.makedirs(cctv_folder, exist_ok=True)
                filename = f"{label}_{cctv_name}_{cv2.getTickCount()}.jpg"
                filepath = os.path.join(cctv_folder, filename)
                cv2.imwrite(filepath, img)
                print(f"Image saved to {filepath}")
                break

    except Exception as e:
        print(f"Error during prediction for {cctv_name}: {str(e)}")
    finally:
        driver.quit()

# 從 CCTV JSON 檔案讀取資料
with open("cctv.json", "r", encoding="utf-8") as f:
    cctv_data = json.load(f)

# 使用多執行緒來處理多個 CCTV
threads = []
for cctv in cctv_data:
    cctv_name = cctv.get("攝影機名稱")
    cctv_url = cctv.get("影像連結網址")
    if cctv_name and cctv_url:
        t = threading.Thread(target=process_cctv, args=(cctv_name, cctv_url))
        threads.append(t)
        t.start()

for t in threads:
    t.join()
