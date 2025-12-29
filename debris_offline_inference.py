#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
離線土石流判釋（彈窗選模型/檔案/輸出）
- 類別順序：0 clear_water, 1 debris-flow, 2 large_rock, 3 muddy_water
- 黃燈：clear_water / muddy_water 的 ΔA 或 ΔC 連續兩幀出現 ±2× 變化
- 紅燈：debris-flow 任一遮罩出現，或「同一顆」large_rock 在連續兩幀位移
- 紅/黃 Hold：1 天；紅燈最高優先不被覆蓋
- 影片支援 --step 跳幀；輸出標註影片/圖 + 逐幀 CSV
"""
import os, cv2, csv, argparse, warnings
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)

# ========= 門檻參數 =========
TWO_FOLD = 2.0              # 兩倍門檻
ROCK_IOU_MATCH = 0.30       # 「同一顆」岩石配對 IoU
ROCK_SHIFT_PROP = 0.50      # 中心位移 / 上幀 bbox 對角線
HOLD_SEC_RED = 86400        # 一天
HOLD_SEC_YEL = 86400        # 一天
EPS = 1e-6

# ========= 類別與顏色 =========
IDX2NAME = {0:'clear_water', 1:'debris-flow', 2:'large_rock', 3:'muddy_water'}
LABELS = ['clear_water','debris-flow','large_rock','muddy_water']
COL = {
    'clear_water':(0,255,255), 'debris-flow':(0,0,255),
    'large_rock':(0,255,0), 'muddy_water':(255,0,0)
}

# ========= GUI：選模型/輸入/輸出 =========
def pick_paths_gui(args):
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk(); root.withdraw()

    if not args.model:
        args.model = filedialog.askopenfilename(
            title='選擇 YOLO11-seg 權重（.pt）',
            filetypes=[('PyTorch Weights','*.pt'), ('All','*.*')]
        )
        if not args.model:
            messagebox.showwarning('中止', '未選擇模型權重'); raise SystemExit

    if not args.input:
        args.input = filedialog.askopenfilename(
            title='選擇輸入影像或影片',
            filetypes=[
                ('Media','*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv'),
                ('Images','*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),
                ('Videos','*.mp4 *.avi *.mov *.mkv'),
                ('All','*.*')
            ]
        )
        if not args.input:
            messagebox.showwarning('中止', '未選擇輸入檔案'); raise SystemExit

    if not args.outdir:
        outdir = filedialog.askdirectory(title='選擇輸出資料夾')
        args.outdir = outdir if outdir else os.path.join(os.getcwd(), 'output')
    os.makedirs(args.outdir, exist_ok=True)
    return args

# ========= 小工具 =========
def overlay_mask(img, mask, color, alpha=0.35):
    if mask is None or not mask.any(): return img
    ov = np.zeros_like(img); ov[mask] = color
    return cv2.addWeighted(img, 1.0, ov, alpha, 0)

def iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    AA = max(0,(a[2]-a[0]))*max(0,(a[3]-a[1])); BB = max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
    return inter/(AA+BB-inter+EPS)

def fold_change(curr, prev):
    if prev <= EPS: return 1.0
    r = curr / max(prev, EPS)
    return r if r >= 1 else 1/r

def attach_panel(frame, rows, alert):
    h,w = frame.shape[:2]; pad=6; lh=22
    panel_h = pad*2 + lh*len(rows)
    panel = np.zeros((panel_h,w,3), np.uint8)
    col0 = {'green':(0,255,0),'yellow':(0,255,255),'red':(0,0,255)}[alert]
    y = pad+16
    for i,t in enumerate(rows):
        col = col0 if i==0 else (255,255,255)
        cv2.putText(panel, t, (pad,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        y += lh
    return np.vstack([frame, panel])

# ========= 單幀推論 =========
def infer_one(model, frame, conf_th, iou_th):
    H,W = frame.shape[:2]
    res = model(frame, conf=conf_th, iou=iou_th, verbose=False)[0]
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None

    confs  = {k:0.0 for k in LABELS}
    areas  = {k:0   for k in LABELS}
    masksU = {k:None for k in LABELS}
    rocks  = []

    for i, bx in enumerate(res.boxes):
        cls_i = int(bx.cls[0]); lbl = IDX2NAME.get(cls_i, str(cls_i))
        if lbl not in COL: continue
        c = float(bx.conf[0]); confs[lbl] = max(confs[lbl], c)
        x1,y1,x2,y2 = map(int, bx.xyxy[0].tolist())

        cv2.rectangle(frame,(x1,y1),(x2,y2),COL[lbl],2)
        cv2.putText(frame,f'{lbl} {c:.2f}',(x1,max(15,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,COL[lbl],2)

        mk = None
        if masks is not None and i < len(masks):
            mk = cv2.resize(masks[i], (W,H), cv2.INTER_NEAREST).astype(bool)
            if mk.any():
                masksU[lbl] = mk if masksU[lbl] is None else (masksU[lbl] | mk)
                areas[lbl]   = int(masksU[lbl].sum())
                frame = overlay_mask(frame, mk, COL[lbl], 0.35)

        if lbl == 'large_rock':
            cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
            diag = ((x2-x1)**2 + (y2-y1)**2)**0.5
            rocks.append({'bbox':[x1,y1,x2,y2], 'center':(cx,cy), 'diag':diag})

    return frame, confs, areas, masksU, rocks

# ========= 燈號判定（Hold 一天、紅優先） =========
class Judge:
    def __init__(self):
        self.prev_area  = {k:0 for k in LABELS}
        self.prev_conf  = {k:0.0 for k in LABELS}
        self.prev_seenA = {k:False for k in LABELS}
        self.prev_seenC = {k:False for k in LABELS}
        self.prev_rocks = []
        self.alert_level = 'green'
        self.alert_exp   = 0
        self.alert_reason= ''

    def step(self, meta, fps, idx):
        # ΔA / ΔC（連續幀才算）
        dA_pct, dC_diff, foldA, foldC = {}, {}, {}, {}
        for k in LABELS:
            a, c = meta['areas'][k], meta['confs'][k]
            seenA = a > 0; seenC = c > 0

            if self.prev_seenA[k] and seenA and self.prev_area[k] > 0:
                dA_pct[k] = (a - self.prev_area[k]) / max(self.prev_area[k], EPS) * 100.0
                foldA[k]  = fold_change(a, self.prev_area[k])
            else:
                dA_pct[k], foldA[k] = 0.0, 1.0

            if self.prev_seenC[k] and seenC and self.prev_conf[k] > 0:
                dC_diff[k] = c - self.prev_conf[k]
                foldC[k]   = fold_change(c, self.prev_conf[k])
            else:
                dC_diff[k], foldC[k] = 0.0, 1.0

            self.prev_area[k], self.prev_conf[k] = a, c
            self.prev_seenA[k], self.prev_seenC[k] = seenA, seenC

        # 紅燈：debris-flow 出現遮罩；或同一顆 large_rock 位移
        raw, reason = 'green',''
        debris_red = meta['areas']['debris-flow'] > 0

        rock_red = False
        if self.prev_rocks and meta['rocks']:
            for pb in self.prev_rocks:
                best = None; best_iou = 0.0
                for cb in meta['rocks']:
                    i = iou(pb['bbox'], cb['bbox'])
                    if i > best_iou: best_iou, best = i, cb
                if best and best_iou >= ROCK_IOU_MATCH and pb['diag'] > EPS:
                    shift = np.hypot(pb['center'][0]-best['center'][0],
                                     pb['center'][1]-best['center'][1]) / pb['diag']
                    if shift >= ROCK_SHIFT_PROP:
                        rock_red = True; break
        self.prev_rocks = meta['rocks']

        if debris_red:
            raw, reason = 'red', 'debris_flow_mask'
        elif rock_red:
            raw, reason = 'red', 'large_rock_moved'
        else:
            # 黃燈：clear / muddy 的 ΔA 或 ΔC 有 ±2×（且連續幀）
            y_ca = (foldA.get('clear_water',1.0) >= TWO_FOLD) or (foldC.get('clear_water',1.0) >= TWO_FOLD)
            y_mu = (foldA.get('muddy_water',1.0) >= TWO_FOLD) or (foldC.get('muddy_water',1.0) >= TWO_FOLD)
            if y_ca or y_mu:
                which = 'both' if (y_ca and y_mu) else ('clear_water' if y_ca else 'muddy_water')
                raw, reason = 'yellow', f'2x_change:{which}'

        # Hold 一天、紅燈優先不覆蓋
        holdR, holdY = int(HOLD_SEC_RED*fps), int(HOLD_SEC_YEL*fps)
        cur, exp, cur_r = self.alert_level, self.alert_exp, self.alert_reason
        if cur == 'red':
            if idx < exp: return 'red', cur_r, dA_pct, dC_diff
            if raw in ('red','yellow'):
                self.alert_level, self.alert_exp, self.alert_reason = 'red', idx+holdR, (reason or cur_r)
                return 'red', self.alert_reason, dA_pct, dC_diff
            self.alert_level, self.alert_exp, self.alert_reason = 'green', 0, ''
            return 'green','', dA_pct, dC_diff

        if cur == 'yellow':
            if idx < exp:
                if raw == 'red':
                    self.alert_level, self.alert_exp, self.alert_reason = 'red', idx+holdR, reason
                    return 'red', reason, dA_pct, dC_diff
                return 'yellow', cur_r, dA_pct, dC_diff
            if raw == 'yellow':
                self.alert_level, self.alert_exp, self.alert_reason = 'yellow', idx+holdY, reason
                return 'yellow', reason, dA_pct, dC_diff
            if raw == 'red':
                self.alert_level, self.alert_exp, self.alert_reason = 'red', idx+holdR, reason
                return 'red', reason, dA_pct, dC_diff
            self.alert_level, self.alert_exp, self.alert_reason = 'green', 0, ''
            return 'green','', dA_pct, dC_diff

        # cur == green
        if raw == 'red':
            self.alert_level, self.alert_exp, self.alert_reason = 'red', idx+holdR, reason
        elif raw == 'yellow':
            self.alert_level, self.alert_exp, self.alert_reason = 'yellow', idx+holdY, reason
        else:
            self.alert_level, self.alert_exp, self.alert_reason = 'green', 0, ''
        return self.alert_level, self.alert_reason, dA_pct, dC_diff

# ========= 影像 / 影片處理 =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  default='', help='yolo11m-seg 權重路徑（留空→彈窗）')
    ap.add_argument('--input',  default='', help='輸入影像或影片（留空→彈窗）')
    ap.add_argument('--outdir', default='', help='輸出資料夾（留空→彈窗）')
    ap.add_argument('--conf',   type=float, default=0.5)
    ap.add_argument('--iou',    type=float, default=0.5)
    ap.add_argument('--step',   type=int,   default=1, help='影片每 N 幀推論一次')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    args = pick_paths_gui(args)
    os.makedirs(args.outdir, exist_ok=True)

    model = YOLO(args.model)
    model.to(args.device)
    torch.backends.cudnn.benchmark = True

    judge = Judge()

    lower = args.input.lower()
    if lower.endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')):
        img = cv2.imread(args.input)
        frm, confs, areas, masksU, rocks = infer_one(model, img, args.conf, args.iou)
        meta = dict(confs=confs, areas=areas, masks=masksU, rocks=rocks)
        alert, reason, dA_pct, dC_diff = judge.step(meta, fps=1.0, idx=1)

        rows = [f'ALERT: {alert.upper()}' + (f' ({reason})' if reason else '')]
        for k in LABELS:
            rows.append(f"{k:<12}  C:{confs[k]:4.2f}  dC:{dC_diff.get(k,0):+4.2f}  dA:{dA_pct.get(k,0):+6.1f}%")
        out = attach_panel(frm, rows, alert)

        base = os.path.splitext(os.path.basename(args.input))[0]
        cv2.imwrite(os.path.join(args.outdir, f'{base}_annotated.jpg'), out)
        with open(os.path.join(args.outdir, f'{base}_result.csv'),'w',newline='',encoding='utf-8') as f:
            cols = ['alert','reason'] + [f'{k}_C' for k in LABELS] + [f'{k}_dC' for k in LABELS] + [f'{k}_dA%' for k in LABELS]
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            row={'alert':alert,'reason':reason}
            row.update({f'{k}_C':round(confs[k],3) for k in LABELS})
            row.update({f'{k}_dC':round(dC_diff.get(k,0.0),3) for k in LABELS})
            row.update({f'{k}_dA%':round(dA_pct.get(k,0.0),2) for k in LABELS})
            w.writerow(row)
        print('✓ 圖片完成')
        return

    # 影片
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>0 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1,args.step)

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_mp4 = os.path.join(args.outdir, f'{base}_annotated.mp4')
    writer = None
    rows_out = []
    pbar = tqdm(total=max(1,total//step), unit='frame')

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            frm, confs, areas, masksU, rocks = infer_one(model, frame, args.conf, args.iou)
            meta = dict(confs=confs, areas=areas, masks=masksU, rocks=rocks)
            alert, reason, dA_pct, dC_diff = judge.step(meta, fps=fps, idx=idx)

            ui = [f'ALERT: {alert.upper()}' + (f' ({reason})' if reason else '')]
            for k in LABELS:
                ui.append(f"{k:<12}  C:{confs[k]:4.2f}  dC:{dC_diff.get(k,0):+4.2f}  dA:{dA_pct.get(k,0):+6.1f}%")
            out = attach_panel(frm, ui, alert)

            if writer is None:
                h,w = out.shape[:2]
                four = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_mp4, four, max(1,int(round(fps/step))), (w,h))
            writer.write(out)

            row={'frame':idx,'alert':alert,'reason':reason}
            row.update({f'{k}_C':round(confs[k],3) for k in LABELS})
            row.update({f'{k}_dC':round(dC_diff.get(k,0.0),3) for k in LABELS})
            row.update({f'{k}_dA%':round(dA_pct.get(k,0.0),2) for k in LABELS})
            rows_out.append(row)
            pbar.update(1)
        idx += 1

    cap.release()
    if writer: writer.release()
    pbar.close()

    with open(os.path.join(args.outdir, f'{base}_result.csv'),'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader(); w.writerows(rows_out)
    print('✓ 影片完成')

if __name__ == '__main__':
    main()
