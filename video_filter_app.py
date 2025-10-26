# -*- coding: utf-8 -*-
# 🎬 Video Filtre Aracı — PySide6 + OpenCV + FFmpeg (YouTube MP4, Realtime, Range Export, No Upscale)
# Özellikler:
# - Canlı önizleme: Parlaklık, Kontrast, Doygunluk, Hue, Gri, Sepya, İnvert, Blur, Opaklık
# - Preset: flat, vivid, cinematic, bw, warm, cool
# - Otomatik modlar: autoBright, autoWhite, autoTone (CLAHE), aiColor, skinTone, scene, recover
# - Başlangıç–Bitiş saniye aralığıyla dışa aktarma (MP4 H.264 + AAC, YouTube uyumlu)
# - FPS: 24/25/30/50/60
# - Çözünürlük: Orijinal, 1080p, 720p, 480p (asla upscale yapmaz)
# - PNG kare kaydı
# - Threaded export + FFmpeg PIPE (rgb24) + even boyut güvenliği

import sys, os, cv2, subprocess, time, tempfile
from pathlib import Path
import numpy as np
import shutil


from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGridLayout, QSlider, QComboBox, QLineEdit, QMessageBox, QProgressBar
)


# ======================= Görüntü İşleme =======================

def clamp255(x): return np.clip(x, 0, 255).astype(np.uint8)

def apply_brightness_contrast(img, b, c):
    alpha = c / 100.0
    beta  = (b - 100) * 2.55
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_saturation_hue(img, s, h):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (s/100.0), 0, 255)
    hsv[:,:,0] = (hsv[:,:,0] + int(h/2)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_grayscale(img, p):
    if p <= 0: return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(g3, p/100.0, img, 1-p/100.0, 0)

def apply_sepia(img, p):
    if p <= 0: return img
    M = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]], np.float32)
    sep = clamp255(cv2.transform(img, M))
    return cv2.addWeighted(sep, p/100.0, img, 1-p/100.0, 0)

def apply_invert(img, p):
    if p <= 0: return img
    inv = 255 - img
    return cv2.addWeighted(inv, p/100.0, img, 1-p/100.0, 0)

def apply_blur(img, k):
    if k <= 0: return img
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k,k), 0)

def apply_opacity(img, p):
    if p >= 100: return img
    return cv2.addWeighted(img, p/100.0, np.zeros_like(img), 1-p/100.0, 0)

def gray_world_white_balance(img):
    b,g,r = cv2.split(img.astype(np.float32))
    avg = (b.mean()+g.mean()+r.mean())/3.0 + 1e-6
    b*=avg/(b.mean()+1e-6); g*=avg/(g.mean()+1e-6); r*=avg/(r.mean()+1e-6)
    return cv2.merge([clamp255(b), clamp255(g), clamp255(r)])

def clahe_auto_tone_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def highlight_shadow_recover(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    lf = l.astype(np.float32)/255.0
    l_out = np.power(lf, 0.9)
    l_out = (np.clip(l_out*255.0, 0, 255)).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_out,a,b]), cv2.COLOR_LAB2BGR)
# === Profesyonel Işık Ayarları ===
def apply_tone_curve(img, shadow=1.0, mid=1.0, highlight=1.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32) / 255.0
    mid_gamma = 1.0 / mid
    lf = np.power(lf, mid_gamma)
    lf = np.clip(lf * highlight, 0, 1)
    lf = np.where(lf < 0.5, lf * shadow, lf)
    l2 = (lf * 255).astype(np.uint8)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# === LUT (.cube) yükleme ===
def load_cube_lut(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    size = 0
    table = []
    for line in lines:
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[-1])
        elif len(line.split()) == 3:
            r, g, b = map(float, line.split())
            table.append([r, g, b])
    return np.array(table).reshape((size, size, size, 3))


def apply_lut(img, lut):
    if lut is None:
        return img
    h, w = img.shape[:2]
    img = img.astype(np.float32) / 255.0
    lut_size = lut.shape[0]
    indices = np.clip((img * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
    out = lut[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

# === LUT (.cube) yükleme ===
def apply_filter_chain(img, st, tone=None, lut=None):
    out = apply_brightness_contrast(img, st['brightness'], st['contrast'])
    out = apply_saturation_hue(out, st['saturate'], st['hue'])
    out = apply_grayscale(out, st['gray'])
    out = apply_sepia(out, st['sepia'])
    out = apply_invert(out, st['invert'])
    out = apply_blur(out, int(st['blur']))
    out = apply_opacity(out, st['opacity'])

    # === Yeni: ToneCurve & LUT ===
    if tone:
        out = apply_tone_curve(out,
                               tone.get('shadow', 1.0),
                               tone.get('mid', 1.0),
                               tone.get('highlight', 1.0))
    if lut is not None:
        out = apply_lut(out, lut)

    return out


def avg_brightness(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())

# ======================= Presetler =======================

PRESETS = {
    "flat":      dict(brightness=100, contrast=95,  saturate=95,  hue=0,   gray=0,  sepia=0, invert=0, blur=0,  opacity=100),
    "vivid":     dict(brightness=105, contrast=120, saturate=135, hue=0,   gray=0,  sepia=0, invert=0, blur=0,  opacity=100),
    "cinematic": dict(brightness=95,  contrast=125, saturate=110, hue=-8,  gray=0,  sepia=10,invert=0, blur=1,  opacity=100),
    "bw":        dict(brightness=100, contrast=120, saturate=0,   hue=0,   gray=100,sepia=0, invert=0, blur=0,  opacity=100),
    "warm":      dict(brightness=102, contrast=108, saturate=115, hue=-10, gray=0,  sepia=8, invert=0, blur=0,  opacity=100),
    "cool":      dict(brightness=102, contrast=108, saturate=110, hue=12,  gray=0,  sepia=0, invert=0, blur=0,  opacity=100),
}

# ======================= Export Thread (FFmpeg PIPE) =======================

def compute_target_size(sel_label, src_w, src_h):
    # Yalnızca küçültmeye izin ver (no-upscale)
    targets = {"Orijinal": src_h, "1080p":1080, "720p":720, "480p":480}
    tgt_h = targets.get(sel_label, src_h)
    if tgt_h >= src_h:  # upscale engelle
        tgt_h = src_h
    # oran koru
    scale = tgt_h / src_h
    tgt_w = int(round(src_w * scale))
    # even zorunluluğu (FFmpeg / H.264 için güvenli)
    if tgt_w % 2: tgt_w -= 1
    if tgt_h % 2: tgt_h -= 1
    tgt_w = max(2, tgt_w)
    tgt_h = max(2, tgt_h)
    return tgt_w, tgt_h

class ExportThread(QThread):
    progress = Signal(int, str)     # (pct, text)
    finished = Signal(str)          # output path
    failed = Signal(str)

    def __init__(self, path_in:str, out_path:str, start_sec:float, end_sec:float,
                 fps_out:int, res_label:str, state:dict, tone=None, lut=None):
        super().__init__()
        self.path_in = path_in
        self.out_path = out_path
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.fps_out = fps_out
        self.res_label = res_label
        self.state = state.copy()
        self.tone = tone or {"shadow":1.0,"mid":1.0,"highlight":1.0}
        self.lut = lut
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        try:
            if not shutil_which("ffmpeg"):
                self.failed.emit("FFmpeg bulunamadı. Lütfen sisteme kurun ve PATH'e ekleyin.")
                return

            cap = cv2.VideoCapture(self.path_in)
            if not cap.isOpened():
                self.failed.emit("Video açılamadı."); return

            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
            start = max(0.0, float(self.start_sec))
            end   = float(self.end_sec) if self.end_sec and self.end_sec > start else duration
            if end <= start:
                end = duration
            # input seek
            cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)

            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tgt_w, tgt_h = compute_target_size(self.res_label, src_w, src_h)

            # FFmpeg pipe (rgb24)
            cmd = [
                "ffmpeg","-y","-loglevel","error",
                "-f","rawvideo","-pix_fmt","rgb24","-s",f"{tgt_w}x{tgt_h}","-r",str(self.fps_out),"-i","-",
                "-ss", str(start), "-to", str(end), "-i", self.path_in,
                "-map","0:v:0","-map","1:a:0?","-shortest",
                "-c:v","libx264","-preset","superfast","-crf","18",
                "-c:a","aac","-b:a","192k",
                self.out_path
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            total_frames_est = int((end - start) * self.fps_out)
            written = 0
            self.progress.emit(0, f"Encode başladı: {tgt_w}x{tgt_h} @ {self.fps_out}fps")

            # ana döngü
            while not self.stop_flag:
                ok, frame = cap.read()
                if not ok: break
                # zamanı kontrol et
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms/1000.0 > end + 0.001:
                    break

                # filtre uygula
                frame = apply_filter_chain(frame, self.state, self.tone, self.lut)
                # downscale (gerekirse)
                if frame.shape[1] != tgt_w or frame.shape[0] != tgt_h:
                    frame = cv2.resize(frame, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                # rgb24 + write
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if proc.poll() is not None:
                    err = proc.stderr.read().decode("utf-8","ignore")
                    self.failed.emit(f"FFmpeg sonlandı:\n{err}")
                    cap.release()
                    return

                try:
                    proc.stdin.write(rgb.tobytes())
                    written += 1
                except Exception as e:
                    cap.release()
                    try: proc.stdin.close()
                    except: pass
                    proc.kill()
                    self.failed.emit(f"PIPE hata: {e}")
                    return

                if total_frames_est>0 and written % max(1, self.fps_out//2) == 0:
                    pct = int(written * 100 / total_frames_est)
                    self.progress.emit(min(pct, 99), f"İşleniyor... {written}/{total_frames_est}")

            cap.release()
            try:
                proc.stdin.flush(); proc.stdin.close()
            except: pass
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode("utf-8","ignore")
                self.failed.emit(f"FFmpeg hata (rc={rc})\n{err}")
                return

            self.progress.emit(100, "Tamamlandı")
            self.finished.emit(self.out_path)

        except Exception as e:
            self.failed.emit(str(e))

def shutil_which(cmd):
    # küçük yardımcı (Windows'ta da çalışır)
    from shutil import which
    return which(cmd)

# ======================= GUI =======================

class VideoFilterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 Video Filtre Aracı — YouTube MP4 • Realtime • Range Export")
        self.resize(1280, 760)
        self.setStyleSheet("""
            QWidget { background:#0b0f14; color:#eaf2fb; font: 500 14px "Segoe UI"; }
            QPushButton { background:#0f1722; border:1px solid #1e2a39; border-radius:10px; padding:8px 10px; }
            QPushButton:hover { border-color:#2a3b51; }
            QGroupBox { border:1px solid #1b2531; border-radius:12px; margin-top:14px; padding:8px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color:#b7c5d6; }
            QSlider::groove:horizontal { background:#1b2531; height:6px; border-radius:3px; }
            QSlider::handle:horizontal { background:#35c37e; width:14px; margin:-6px 0; border-radius:7px; }
            QComboBox, QLineEdit, QProgressBar { background:#0f1722; border:1px solid #1e2a39; border-radius:8px; padding:6px; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #35c37e, stop:1 #2aa3ff); border-radius:6px; }
        """)

        # durum
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.video_path = None
        self.frame = None
        self.paused = True
        self.fps_src = 30.0
        self.total_frames = 0
        self.cur_frame_idx = 0
        self.exporter = None

        self.state = PRESETS["flat"].copy()
        self.tone = {"shadow": 1.0, "mid": 1.0, "highlight": 1.0}
        self.lut_data = None

        self.build_ui()

    def build_ui(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()

        # ==== SAĞ: Önizleme ====
        self.video_label = QLabel("Önizleme")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 405)
        self.video_label.setStyleSheet("background:#000; border:1px solid #1b2531; border-radius:12px;")
        right.addWidget(self.video_label, 1)

        # ==== SOL: Kontroller ====

        # Dosya / oynatma
        row = QHBoxLayout()
        self.btn_open = QPushButton("📂 Video Aç"); self.btn_open.clicked.connect(self.open_video)
        self.btn_play = QPushButton("▶︎ Oynat / Duraklat"); self.btn_play.clicked.connect(self.toggle_play)
        self.btn_png  = QPushButton("📸 PNG Kaydet"); self.btn_png.clicked.connect(self.save_png)
        row.addWidget(self.btn_open); row.addWidget(self.btn_play); row.addWidget(self.btn_png)
        left.addLayout(row)

        # Presetler
        presetBox = QGroupBox("Hazır Ayarlar")
        pLay = QHBoxLayout()
        for name in ["flat","vivid","cinematic","bw","warm","cool"]:
            b = QPushButton(name.capitalize())
            b.clicked.connect(lambda _, n=name: self.apply_preset(n))
            pLay.addWidget(b)
        presetBox.setLayout(pLay)
        left.addWidget(presetBox)

        # Otomatik modlar
        autoBox = QGroupBox("Otomatik Modlar")
        aLay = QGridLayout()
        modes = [("Oto Parlak/Kontrast","autoBright"),
                 ("Oto Beyaz Dengesi","autoWhite"),
                 ("Auto Tone (CLAHE)","autoTone"),
                 ("AI Color Enhance","aiColor"),
                 ("Cilt Düzeltici","skinTone"),
                 ("Sahne Algılama","scene"),
                 ("Işık/Gölge Kurtarma","recover")]
        for i,(label,key) in enumerate(modes):
            b = QPushButton(label); b.clicked.connect(lambda _, k=key: self.apply_auto_mode(k))
            aLay.addWidget(b, i//2, i%2)
        autoBox.setLayout(aLay)
        left.addWidget(autoBox)

        # Sliderlar
        sliders = QGroupBox("Filtreler")
        grid = QGridLayout()

        def add_slider(row, text, key, mn, mx, val, step=1):
            lab = QLabel(f"{text}: {val}")
            s = QSlider(Qt.Horizontal); s.setRange(mn,mx); s.setSingleStep(step); s.setValue(val)
            s.valueChanged.connect(lambda v,k=key,l=lab,t=text: self.on_slider(k, v, l, t))
            grid.addWidget(lab,row,0); grid.addWidget(s,row,1); return s, lab

        self.sld={}
        self.sld['brightness'], self.lab_b  = add_slider(0,"Parlaklık","brightness",0,300,self.state['brightness'])
        self.sld['contrast'],   self.lab_c  = add_slider(1,"Kontrast","contrast",0,300,self.state['contrast'])
        self.sld['saturate'],   self.lab_s  = add_slider(2,"Doygunluk","saturate",0,300,self.state['saturate'])
        self.sld['hue'],        self.lab_h  = add_slider(3,"Hue (°)","hue",-180,180,self.state['hue'])
        self.sld['gray'],       self.lab_g  = add_slider(4,"Grayscale %","gray",0,100,self.state['gray'])
        self.sld['sepia'],      self.lab_sp = add_slider(5,"Sepya %","sepia",0,100,self.state['sepia'])
        self.sld['invert'],     self.lab_iv = add_slider(6,"Invert %","invert",0,100,self.state['invert'])
        self.sld['blur'],       self.lab_bl = add_slider(7,"Blur px","blur",0,20,int(self.state['blur']))
        self.sld['opacity'],    self.lab_op = add_slider(8,"Opacity %","opacity",10,100,self.state['opacity'])
        # === Yeni Işık Ayarları (Tone Curve) ===
        self.sld['shadow'], _ = add_slider(9, "Shadow", "shadow", 50, 150, 100)
        self.sld['mid'], _ = add_slider(10, "Midtone", "mid", 50, 150, 100)
        self.sld['highlight'], _ = add_slider(11, "Highlight", "highlight", 50, 150, 100)
        
        # === LUT Butonları ===
        self.btn_lut = QPushButton("🎨 LUT (.cube) Yükle")
        self.btn_lut_clear = QPushButton("🗑️ LUT Kaldır")
        self.btn_lut.clicked.connect(self.load_lut)
        self.btn_lut_clear.clicked.connect(self.clear_lut)
        grid.addWidget(self.btn_lut, 12, 0)
        grid.addWidget(self.btn_lut_clear, 12, 1)

        sliders.setLayout(grid)
        left.addWidget(sliders)

        # Çıktı ayarları (FPS/Resolution/Aralık)
        outBox = QGroupBox("Dışa Aktarma Ayarları")
        h = QGridLayout()
        self.cmb_fps = QComboBox(); self.cmb_fps.addItems(["24","25","30","50","60"]); self.cmb_fps.setCurrentText("30")
        self.cmb_res = QComboBox(); self.cmb_res.addItems(["Orijinal","1080p","720p","480p"])
        self.in_start = QLineEdit("0"); self.in_start.setPlaceholderText("Başlangıç sn (örn: 0)")
        self.in_end   = QLineEdit("");  self.in_end.setPlaceholderText("Bitiş sn (boş=son)")
        h.addWidget(QLabel("FPS"),0,0); h.addWidget(self.cmb_fps,0,1)
        h.addWidget(QLabel("Çözünürlük"),1,0); h.addWidget(self.cmb_res,1,1)
        h.addWidget(QLabel("Başlangıç (sn)"),2,0); h.addWidget(self.in_start,2,1)
        h.addWidget(QLabel("Bitiş (sn)"),3,0); h.addWidget(self.in_end,3,1)
        outBox.setLayout(h)
        left.addWidget(outBox)

        # Export kontrol
        row2 = QHBoxLayout()
        self.btn_export = QPushButton("⏺️ Dışa Aktar (MP4)")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_reset  = QPushButton("Sıfırla (Flat)")
        self.btn_reset.clicked.connect(lambda: self.apply_preset("flat"))
        row2.addWidget(self.btn_export); row2.addWidget(self.btn_reset)
        left.addLayout(row2)

        # Durum/Bar
        self.prog = QProgressBar(); self.prog.setRange(0,100); self.prog.setValue(0)
        self.lbl_status = QLabel("Hazır")
        left.addWidget(self.prog); left.addWidget(self.lbl_status)

        root.addLayout(left, 1); root.addLayout(right, 2)

    # ==== Slider/Preset/Auto ====
    def on_slider(self, key, value, lab, text):
        if key in ['shadow', 'mid', 'highlight']:
            self.tone[key] = value / 100.0
        else:
            self.state[key] = int(value) if key != 'blur' else float(value)

        lab.setText(f"{text}: {value}")
        if self.frame is not None:
            out = apply_filter_chain(self.frame.copy(), self.state, self.tone, self.lut_data)
            self.show_frame(out)

    def apply_preset(self, name):
        st = PRESETS.get(name, PRESETS["flat"])
        self.state.update(st)
        for k,v in st.items():
            if k in self.sld:
                self.sld[k].blockSignals(True)
                self.sld[k].setValue(int(v))
                self.sld[k].blockSignals(False)
        self.lbl_status.setText(f"Preset: {name}")
        if self.frame is not None:
            self.show_frame(apply_filter_chain(self.frame.copy(), self.state, self.tone, self.lut_data))

    def apply_auto_mode(self, mode):
        if self.frame is None:
            QMessageBox.warning(self, "Uyarı","Önce bir video aç/oynat!")
            return
        frm = self.frame.copy()
        a = avg_brightness(frm)
        if mode == "autoBright":
            self.state['brightness'] = int(np.clip(100 + ((128-a)/2), 50, 200))
            self.state['contrast']   = int(np.clip(100 + (abs(128-a)/1.5), 80, 200))
        elif mode == "autoWhite":
            f2 = gray_world_white_balance(frm)
            self.state['hue'] = int(np.clip((f2.mean()-frm.mean())/3, -15, 15))
        elif mode == "autoTone":
            auto = clahe_auto_tone_bgr(frm)
            d = float(auto.mean() - frm.mean())
            self.state['brightness'] = int(np.clip(100 + d/2, 50, 200))
            self.state['contrast']   = int(np.clip(110 + abs(d)/3, 80, 200))
        elif mode == "aiColor":
            self.state['saturate']=130; self.state['contrast']=120; self.state['brightness']=105; self.state['hue']=-3
        elif mode == "skinTone":
            self.state['saturate']=115; self.state['hue']=-8; self.state['sepia']=5
        elif mode == "scene":
            if a < 70:      self.apply_preset("warm")
            elif a < 120:   self.apply_preset("cinematic")
            else:           self.apply_preset("vivid")
            self.lbl_status.setText("Sahne algılama uygulandı")
            return
        elif mode == "recover":
            self.state['contrast']=115; self.state['brightness']=108; self.state['saturate']=110

        for k,v in self.state.items():
            if k in self.sld:
                self.sld[k].blockSignals(True)
                self.sld[k].setValue(int(v))
                self.sld[k].blockSignals(False)
        self.lbl_status.setText(f"Otomatik: {mode}")
        if self.frame is not None:
            self.show_frame(apply_filter_chain(self.frame.copy(), self.state))
    def load_lut(self):
        p, _ = QFileDialog.getOpenFileName(self, "LUT (.cube) Seç", "", "LUT Files (*.cube)")
        if not p: return
        try:
            self.lut_data = load_cube_lut(p)
            self.lbl_status.setText(f"LUT yüklendi: {Path(p).name}")
        except Exception as e:
            QMessageBox.critical(self, "LUT Hatası", str(e))
    
    def clear_lut(self):
        self.lut_data = None
        self.lbl_status.setText("LUT kaldırıldı")
    
    # ==== Video ====
    def open_video(self):
        p,_ = QFileDialog.getOpenFileName(self,"Video Seç","","Video Files (*.mp4 *.mov *.mkv *.avi *.webm)")
        if not p: return
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(p)
        if not self.cap.isOpened():
            QMessageBox.critical(self,"Hata","Video açılamadı!"); return
        self.video_path = p
        self.fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, fr = self.cap.read()
        if ok:
            self.frame = fr
            self.show_frame(fr)
        self.paused = False
        self.timer.start(int(1000/min(self.fps_src, 60)))
        self.lbl_status.setText(f"Yüklendi: {Path(p).name}")

    def toggle_play(self):
        if not self.cap: return
        self.paused = not self.paused
        if not self.paused:
            self.timer.start(int(1000/min(self.fps_src, 60)))
        self.lbl_status.setText("Oynatılıyor" if not self.paused else "Duraklatıldı")

    def update_preview(self):
        if not self.cap or self.paused: return
        ok, frame = self.cap.read()
        if not ok:
            # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); return
        self.frame = frame
        out = apply_filter_chain(self.frame.copy(), self.state, self.tone, self.lut_data)
        self.show_frame(out)

    def show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ==== PNG ====
    def save_png(self):
        if self.frame is None:
            QMessageBox.warning(self,"Uyarı","Önce video oynat!"); return
        out = apply_filter_chain(self.frame.copy(), self.state, self.tone, self.lut_data)
        p,_ = QFileDialog.getSaveFileName(self,"PNG Kaydet","kare.png","PNG (*.png)")
        if not p: return
        cv2.imwrite(p, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        self.lbl_status.setText(f"PNG kaydedildi: {Path(p).name}")

    # ==== EXPORT ====
    def export_video(self):
        if not self.video_path:
            QMessageBox.warning(self,"Uyarı","Önce bir video seç!"); return
        out_path,_ = QFileDialog.getSaveFileName(self,"Dışa Aktar","filtreli-video.mp4","MP4 Video (*.mp4)")
        if not out_path: return

        try:
            start = float(self.in_start.text() or 0)
        except: start = 0.0
        try:
            end = float(self.in_end.text()) if self.in_end.text().strip() else 0.0
        except: end = 0.0

        fps_out = int(self.cmb_fps.currentText())
        res_label = self.cmb_res.currentText()

        # thread başlat
        self.exporter = ExportThread(self.video_path, out_path, start, end, fps_out, res_label, self.state, self.tone, self.lut_data)
        self.exporter.progress.connect(self.on_export_progress)
        self.exporter.finished.connect(self.on_export_finished)
        self.exporter.failed.connect(self.on_export_failed)
        self.exporter.start()
        self.lbl_status.setText("Dışa aktarma başladı…")

    def on_export_progress(self, pct, txt):
        self.prog.setValue(pct); self.lbl_status.setText(txt)

    def on_export_finished(self, path):
        self.prog.setValue(100)
        self.lbl_status.setText(f"Bitti: {Path(path).name}")
        self.exporter = None

    def on_export_failed(self, msg):
        QMessageBox.critical(self,"Export Hatası", msg)
        self.lbl_status.setText("Dışa aktarma hatası")
        self.exporter = None

# ======================= MAIN =======================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = VideoFilterApp()
    w.show()
    sys.exit(app.exec())
