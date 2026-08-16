# 🌊 土石流即時監測系統 | Debris Flow Monitoring System

以 YOLO11-seg 對全臺土石流觀測站的 CCTV 影像做即時判釋，並提供地圖儀表板與值班監控台。

Real-time debris flow detection over Taiwan's debris-flow CCTV network, with a
map dashboard and a duty-room operations console.

---

# 說明

## 這是什麼

抓取土石流觀測站的 CCTV 畫面 → 用 YOLO11-seg 做影像分割判釋 → 依判釋結果決定紅／黃／綠燈 → 推到網頁儀表板。

判釋類別分成四類：清水（clear_water）、混濁水（muddy_water）、土石流（debris_flow）、大型岩塊（large_rock）。

## 燈號怎麼判

| 燈號 | 判定條件 |
|:---:|---|
| 🟢 綠燈 | 有清水，或沒有混濁水也沒有土石流 |
| 🟡 黃燈 | 混濁水佔優勢（比例或信心值高過清水） |
| 🔴 紅燈 | **已經是黃燈**時，再偵測到土石流或岩塊移動 |

紅燈會 hold 一天（`HOLD_RED_SEC`），期間不會被其他判定蓋掉。

**紅燈的「兩段式」是刻意設計** 從綠燈直接看到土石流時，這一輪只會先進黃燈，要連續兩輪才會轉紅。目的是避免單一張雜訊畫面就觸發長達一天的紅燈鎖定，跟 `MOVE_CONSEC=3`、`MIN_YELLOW_SEC`、`YELLOW_CLEAR_SEC` 是同一套遲滯設計。

離線版 `debris_offline_inference.py` 則是「一偵測到就轉紅」，兩邊判定邏輯不同是已知且刻意的差異，請不要為了「一致」而改掉其中一邊。

## 系統架構

```
土石流觀測站 CCTV
      │  Selenium 取得影像來源（支援 iframe 與 data:image base64）
      ▼
yolo11m_seg_detectfromcctv_V3.PY   ← 主程式，多執行緒偵測
      │  寫出
      ├─► camera_status.json   各站最新燈號／比例／信心值／歷史影像
      ├─► alert_state.json     紅燈鎖定狀態（跨重啟存活）
      ├─► saved_images/        判釋圖與原圖
      └─► logs/                metrics/<站>.csv 逐幀紀錄、events.jsonl 事件
      │
      ├─► server.py            FastAPI，供網頁讀取；並記錄處置到 logs/acks.jsonl
      │        └─► web_gis/console.html   值班監控台
      │
      └─► sync_to_cloud.py     （選用）同步到 Render 雲端
```

## 檔案結構

```
yolo11m_seg_detectfromcctv_V3.PY  # 主程式：即時 CCTV 判釋
server.py                         # FastAPI 伺服器 / API
sync_to_cloud.py                  # 將本機判釋結果同步到 Render
debris_offline_inference.py       # 離線影像／影片判釋（有圖形介面）

web_gis/console.html              # 值班監控台（唯一的網頁介面）
web_gis/demo/demo-data.json       # 公開展示版的虛構資料
saved_images/demo/                # 公開展示版的合成影像

weights/yolo11m-seg.pt            # YOLO11-seg 權重（未進 git，檔案太大）
cctv.json                         # 攝影機清單（未進 git，屬本機資料）
camera_status.json                # 各站即時狀態（未進 git，由主程式產生）
alert_state.json                  # 紅燈鎖定狀態（未進 git，由主程式產生）
logs/                             # 逐幀紀錄、事件、處置紀錄（未進 git）

requirements.txt                  # 網頁儀表板相依套件（Render 安裝這份）
requirements-detector.txt         # 本機判釋相依套件（YOLO、Selenium 等）
docs/                             # 訓練紀錄
```

舊的 `web_gis/index.html`（地圖儀表板）與 `web_gis/viewer.html`（單站檢視）已由
`console.html`取代並刪除。需要時可從 git 歷史取回：
`git show <刪除前的 commit>:web_gis/index.html`

## 值班監控台

`web_gis/console.html`，桌面三欄、手機單欄（底部頁籤：警報／地圖／詳情，預設落在警報）。

**燈號之外，介面另外處理三件 `camera_status.json` 本身講不出來的事：**

| 問題 | 處理方式 |
|---|---|
| 紅燈 hold 24 小時，`timestamp` 每輪更新，所以「今早發生的」和「現在正在發生的」長得一樣 | 偵測端另寫 `redSince` / `redHoldUntil`；介面分成「紅燈警報（現正發生）」與「紅燈鎖定中」兩區，只有前者會閃爍 |
| 一台掛掉的相機會停在最後一次的綠燈上 | 兩個獨立訊號：偵測端明說的抓取失敗（`download_fail`），以及回報時間超過該站自身節奏（`reportInterval`）的倍數。**全部測站同時無回報時判定為偵測程式未運行，只顯示一則系統訊息，不產生 63 則故障** |
| 誰處理了哪一則警報、為什麼 | 確認警報必須署名並選擇原因，寫入 `logs/acks.jsonl`（只增不刪）。確認以事件為單位，同一測站發生新的紅燈時舊確認自動失效 |

標記為「誤判」的處置會連同當下影像路徑一起記錄——那是已標註的偽陽性樣本，之後調模型或調閾值時可直接使用。

**「解除鎖定」不會解除偵測端的 24 小時 hold。** 偵測程式只在啟動時讀狀態檔，伺服器寫回去不會即時生效；讓一個網頁點擊靜默解除安全機制方向也不對。解除只影響介面呈現，並留下紀錄。

## 公開展示版

```
web_gis/console.html?demo=1
```

讀 `web_gis/demo/demo-data.json`，**完全不呼叫 `/api/*`**，所以可以直接放上 GitHub Pages
之類的靜態空間，不需要後端。

展示資料全部是虛構的：測站名稱、行政區、座標都是編造的（示範縣、範例鄉…），影像是
合成的並燒上「DEMO IMAGE」字樣。這是刻意的——一個看起來像官方即時監測、指名某條真實
溪流正在發生土石流的頁面，被截圖轉傳出去的後果不對稱。頁面頂端有不可關閉的橫幅，瀏覽器
標題也帶「【展示版】」。

## 安裝

### 1. 網頁儀表板

```bash
pip install -r requirements.txt
python server.py
# 開 http://localhost:8000/   （根路徑即值班監控台）
```

### API

| 端點 | 用途 |
|---|---|
| `GET /camera_status.json` | 各站即時狀態 |
| `GET /cctv.json` | 攝影機清單 |
| `GET /api/metrics/{站名}` | 該站近 24 小時逐幀紀錄（讀 `logs/metrics/<站>.csv`，超量自動降採樣但保留燈號轉換點） |
| `GET /api/events` | 燈號轉換歷史 + 各站最近一次影像抓取失敗（尾端讀取 `events.jsonl`，跳過壞行） |
| `GET /api/acks` | 目前各站的處置狀態 |
| `POST /api/acks` | 記錄確認／取消／解除，需帶 `by`、`reason` |

### 2. 本機判釋（需要 GPU 比較實際）

```bash
pip install -r requirements-detector.txt
```

⚠️ **torch 請自己裝，不要讓 pip 順手裝掉。** PyPI 上的 torch 常常是 CPU-only 版本，一不小心就會把能用的 CUDA 版蓋掉，變成 `torch.cuda.is_available()` 回傳 `False`，還會跟 torchvision 版本對不起來。請對著你的 CUDA 版本裝：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. 攝影機清單

主程式需要 `cctv.json`，格式如下（**這份檔案不在 git 上，需自行準備**）：

```json
[
  {
    "攝影機名稱": "範例攝影機",
    "縣市": "南投縣",
    "行政區": "信義鄉",
    "架設或拍攝地點": "某某溪上游",
    "影像連結網址": "https://example.com/cam.jpg",
    "經度": 120.9,
    "緯度": 23.7
  }
]
```

## 執行

```bash
python yolo11m_seg_detectfromcctv_V3.PY          # 主程式（產生 camera_status.json）
python server.py                                 # 網頁伺服器
python sync_to_cloud.py                          # （選用）同步到雲端
```

常用參數：

```bash
python yolo11m_seg_detectfromcctv_V3.PY --focus 集來下游攝影機   # 只看特定測站
python yolo11m_seg_detectfromcctv_V3.PY --threads 8            # 調整並行數
python yolo11m_seg_detectfromcctv_V3.PY --interval 2.0         # 調整偵測間隔
python yolo11m_seg_detectfromcctv_V3.PY --conf 0.6             # 調整信心門檻
```

紅燈／黃燈的測站會自動提高偵測頻率（紅 > 黃 > 綠）。

## ChromeDriver 不用自己準備

專案裡**沒有**附 chromedriver。Chrome 會自動更新，附在專案裡的 driver 遲早版本對不上；而版本一對不上，過去會讓所有 worker 在啟動當下就死光，而且主程式完全不會察覺（看起來還活著，實際上什麼都沒在監看）。

現在 `create_driver()` 會交給 Selenium Manager（selenium >= 4.6 內建）自動抓對應版本並快取。你若真的要放自己的 driver 在 `chromedriver-win64/chromedriver.exe`，程式會優先用它，失敗再退回 Selenium Manager。

## 部署到 Render

1. Fork 這個 repo
2. 到 [Render.com](https://render.com) 建立 Web Service
3. Render 會自動讀取 `render.yaml`
4. 環境變數設定 `SYNC_API_KEY`（給 `sync_to_cloud.py` 上傳用）

雲端只跑儀表板，不跑判釋 —— 判釋在本機做完再同步上去。

## 注意事項

- `cctv.json`、`camera_status.json`、`.env`、`secrets.env` 都在 `.gitignore` 內，**不要**把它們推上公開 repo。
- 影像網址一律存**相對路徑**（`/saved_images/...`）。存絕對網址會導致換埠號或從其他機器開就整片破圖。
- `camera_status.json` 由多執行緒共寫，程式內以鎖序列化並用 `os.replace` 原子性置換，讀取端不會讀到寫到一半的檔案。

## 已知事項

模型 `weights/yolo11m-seg.pt` 實際輸出 5 類：`debris-flow`、`level`、`muddy`、`riverbond`、`rock`。其中 4 類會經 `CLASS_MAP` 對應到系統的四類，但 **`level` 沒有對應，會被直接略過**（在畫面中其實經常被偵測到）。若 `level` 對判釋有意義，需要另外決定要怎麼併入燈號邏輯。

---

# English

## What this is

Pulls frames from Taiwan's debris-flow CCTV stations → runs YOLO11-seg
segmentation → derives a red/yellow/green alert level → serves it to a web
dashboard.

Four classes: `clear_water`, `muddy_water`, `debris_flow`, `large_rock`.

## Alert logic

| Level | Condition |
|:---:|---|
| 🟢 green | clear water present, or neither muddy water nor debris flow |
| 🟡 yellow | muddy water dominates clear water (by area proportion or confidence) |
| 🔴 red | debris flow or rock movement detected **while already yellow** |

Red is held for a day (`HOLD_RED_SEC`) and is not overwritten during that window.

⚠️ **The two-step path to red is deliberate, not a bug.** A debris flow seen from
a green baseline only raises yellow on that cycle; it takes two consecutive
cycles to reach red. This prevents a single noisy frame from latching a
24-hour red alert, and is consistent with `MOVE_CONSEC=3`, `MIN_YELLOW_SEC` and
`YELLOW_CLEAR_SEC`.

The offline script `debris_offline_inference.py` escalates to red immediately on
any debris-flow mask. The two differ on purpose — please don't "fix" one to
match the other.

## Architecture

```
Debris-flow CCTV stations
      │  Selenium resolves the image source (handles iframes and data: base64)
      ▼
yolo11m_seg_detectfromcctv_V3.PY   ← main detector, multi-threaded
      │  writes
      ├─► camera_status.json   per-station alert / proportion / confidence / history
      ├─► alert_state.json     red-hold state, survives a detector restart
      ├─► saved_images/        annotated + raw frames
      └─► logs/                metrics/<cam>.csv per frame, events.jsonl
      │
      ├─► server.py            FastAPI; also records handling to logs/acks.jsonl
      │        └─► web_gis/console.html   operations console
      │
      └─► sync_to_cloud.py     (optional) mirror to Render
```

## The console

`web_gis/console.html` is the only web page. Three columns on a desktop, one
column with bottom tabs on a phone.

Beyond the traffic light it handles three things `camera_status.json` cannot
express on its own:

- **A red alert is held for 24 hours, and `timestamp` refreshes every cycle.**
  A debris flow from this morning therefore looked identical to one starting
  now. The detector now also writes `redSince` and `redHoldUntil`, and the
  console separates 現正發生 from 鎖定中. Only the former pulses.
- **A dead camera used to sit on its last green reading.** Health now comes
  from two independent signals: the detector's own `download_fail` events, and
  silence measured against each station's own `reportInterval`. If *every*
  station goes quiet at once the detector itself is presumed down and the board
  says so once, rather than raising 63 separate faults.
- **Who handled which alert, and why.** Acknowledging requires a name and a
  reason, appended to `logs/acks.jsonl`. Acknowledgements are scoped to one
  alert episode, so a new red at the same camera is never pre-silenced by the
  previous one. Reasons in the false-positive family keep the frame paths —
  labelled negatives for the next round of model tuning.

Releasing a held red **does not** clear the detector's 24-hour hold. The
detector reads its state file only at startup, and a web click that silently
disarms a safety mechanism is the wrong shape regardless. The release changes
what the board shows and leaves a record.

## Public demo

`web_gis/console.html?demo=1` reads `web_gis/demo/demo-data.json` and never
calls `/api/*`, so it can be hosted on GitHub Pages with no backend.

Every station in it is invented — fictional names, districts and coordinates —
and the frames are synthetic with "DEMO IMAGE" burned in. A page that looks
like live monitoring and names a real river as having a debris flow can be
screenshotted and forwarded; the consequences of that are not symmetric. A
non-dismissible banner and a 【展示版】 title prefix travel with it.

## Setup

Dashboard:

```bash
pip install -r requirements.txt
python server.py
# open http://localhost:8000/   (the root path is the console)
```

Detector:

```bash
pip install -r requirements-detector.txt
```

⚠️ **Install torch yourself; don't let pip resolve it as a side effect.** PyPI
wheels are often CPU-only and will silently replace a working CUDA build,
leaving `torch.cuda.is_available()` False and desyncing torchvision:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The detector needs a `cctv.json` camera list, which is **not in this repo** — see
the schema in the Chinese section above.

## Running

```bash
python yolo11m_seg_detectfromcctv_V3.PY   # produces camera_status.json
python server.py                          # web server
python sync_to_cloud.py                   # (optional) cloud sync
```

Useful flags: `--focus <camera>`, `--threads N`, `--interval SEC`, `--conf F`.
Red/yellow stations are polled more frequently than green ones automatically.

## ChromeDriver

**Not bundled.** Chrome auto-updates and a checked-in driver goes stale; a
mismatch used to kill every worker thread at startup while the main loop
happily reported nothing wrong. `create_driver()` now falls back to Selenium
Manager (selenium >= 4.6), which fetches and caches a matching driver. A local
`chromedriver-win64/chromedriver.exe` is still tried first if present.

## Deploying to Render

1. Fork this repository
2. Create a Web Service on [Render.com](https://render.com)
3. `render.yaml` is auto-detected
4. Set `SYNC_API_KEY` for `sync_to_cloud.py` uploads

The cloud instance only serves the dashboard — inference runs locally and the
results are synced up.

## Notes

- `cctv.json`, `camera_status.json`, `.env` and `secrets.env` are gitignored.
  Keep them out of public repositories.
- Image URLs are stored **relative** (`/saved_images/...`). Absolute URLs break
  as soon as you change port or view from another machine.
- `camera_status.json` is written by several worker threads; writes are
  serialized under a lock and swapped in with `os.replace`, so readers never
  observe a half-written file.

## Known issue

`weights/yolo11m-seg.pt` emits five classes: `debris-flow`, `level`, `muddy`,
`riverbond`, `rock`. Four are mapped to the system's four categories via
`CLASS_MAP`, but **`level` is unmapped and silently discarded**, despite being
detected frequently. If `level` is meaningful for alerting, it needs an explicit
decision on how to fold it into the logic.
