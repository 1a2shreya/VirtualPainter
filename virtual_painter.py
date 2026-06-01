#!/usr/bin/env python3
"""
Virtual Painter Pro v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES:
  ✅ Two-finger selection: STABLE_FRAMES 4→3, separate
     cooldowns for toolbar/shape/emoji, shape commits
     at last-draw position (accurate), permissive ring/
     pinky detection
  ✅ Voice: word-level fuzzy matching, more commands,
     periodic noise recalibration, exponential backoff,
     animated listening indicator

NEW UI:
  ✨ Gradient toolbar with rounded swatches & tool btns
  ✨ Bottom status bar  (mode · color · brush · FPS)
  ✨ Voice status badge with pulse animation
  ✨ Confirmation flash on toolbar selection
  ✨ Improved brush cursor & eraser ring
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gestures  ☝ 1 finger  → DRAW
          ✌ 2 fingers → SELECT / confirm shape
          🤏 pinch    → resize brush
Keys      Z=Undo  Y=Redo  S=Save  C=Clear  T=Text
          1-8=Emoji   Q/ESC=Quit
Voice     "red" "clear" "pen" "circle" "undo" "bigger" …
"""

import cv2, numpy as np, math, time, os, threading, urllib.request
from datetime import datetime
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import speech_recognition as sr
    VOICE_OK = True
except ImportError:
    VOICE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H   = 1280, 720
TB_H           = 150          # toolbar height px
SB_H           = 44           # bottom status-bar height px
STABLE_FRAMES  = 3            # FIX ↓ was 4 — faster gesture confirm
MIN_BRUSH, MAX_BRUSH = 2, 60
UNDO_LIMIT     = 40

# Per-action cooldowns (seconds)  — FIX: was one shared click_t
TB_COOLDOWN    = 0.35
SHAPE_COOLDOWN = 0.40
EMOJI_COOLDOWN = 0.50
PICKER_COOLDOWN= 0.10

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── UI color palette (BGR) ───────────────────────────────────────────────────
C_BG       = (30, 28, 36)
C_BG2      = (20, 18, 26)
C_ACCENT   = (30, 190, 255)    # warm gold
C_SELECT   = (60, 200, 255)
C_ACTIVE   = (55, 230, 100)    # green
C_HOVER    = (0,  230, 240)    # cyan
C_TEXT     = (220, 218, 232)
C_MUTED    = (110, 108, 128)
C_BORDER   = (60,  58,  78)
C_SEP      = (48,  46,  62)
C_TOOL_BG  = (42,  40,  55)
C_TOOL_ACT = (22,  75,  38)    # dark green
C_SB       = (16,  15,  20)

# ── Drawing palette (BGR) ────────────────────────────────────────────────────
PALETTE = [
    ("Red",     ( 0,   0, 255)), ("Orange",  ( 0, 130, 255)), ("Yellow", ( 0, 220, 255)),
    ("Lime",    ( 0, 255,   0)), ("Green",   ( 0, 180,   0)), ("Teal",   (80, 200,  80)),
    ("Cyan",    (255, 220,  0)), ("Sky",     (255, 160,  80)), ("Blue",  (255,  30,  30)),
    ("Navy",    (140,  20,  20)), ("Purple", (200,   0, 190)), ("Pink",  (180, 100, 255)),
    ("Magenta", (255,   0, 200)), ("Brown",  ( 40,  80, 150)), ("Tan",   (100, 170, 200)),
    ("White",   (255, 255, 255)), ("Gray",   (160, 160, 160)), ("Black", ( 10,  10,  10)),
]

BRUSH_SIZES = [3, 7, 14, 24, 36]
TOOLS       = ['pen','eraser','line','rect','circle','fill','text','emoji']
TOOL_LABEL  = {
    'pen':    'PEN',   'eraser': 'ERASE',  'line':   'LINE',
    'rect':   'RECT',  'circle': 'CIRC',   'fill':   'FILL',
    'text':   'TEXT',  'emoji':  'EMO',
}
EMOJIS = ['😀','❤️','⭐','🚀','🎨','✨','🎉','🔥']

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

# ── Voice command map — single-word keys for fuzzy matching ──────────────────
VOICE_MAP = {
    # Actions
    'clear':'clear',   'wipe':'clear',    'reset':'clear',
    'save':'save',     'export':'save',   'capture':'save',
    'undo':'undo',     'back':'undo',
    'redo':'redo',     'forward':'redo',
    # Brush size
    'bigger':'brush_up',    'larger':'brush_up',    'increase':'brush_up',
    'smaller':'brush_down', 'decrease':'brush_down', 'tiny':'brush_down',
    # Tools
    'pen':('tool','pen'),       'pencil':('tool','pen'),   'draw':('tool','pen'),
    'eraser':('tool','eraser'), 'erase':('tool','eraser'), 'rubber':('tool','eraser'),
    'line':('tool','line'),     'straight':('tool','line'),
    'rectangle':('tool','rect'),'rect':('tool','rect'),    'square':('tool','rect'),
    'circle':('tool','circle'), 'oval':('tool','circle'),  'round':('tool','circle'),
    'fill':('tool','fill'),     'bucket':('tool','fill'),  'paint':('tool','fill'),
    'text':('tool','text'),     'write':('tool','text'),   'type':('tool','text'),
    'emoji':('tool','emoji'),   'sticker':('tool','emoji'),
    # Colors
    'red':     ('color',(  0,  0,255)), 'orange':('color',(  0,130,255)),
    'yellow':  ('color',(  0,220,255)), 'lime':  ('color',(  0,255,  0)),
    'green':   ('color',(  0,200,  0)), 'teal':  ('color',( 80,200, 80)),
    'cyan':    ('color',(255,220,  0)), 'sky':   ('color',(255,160, 80)),
    'blue':    ('color',(255,  0,  0)), 'navy':  ('color',(140, 20, 20)),
    'purple':  ('color',(200,  0,190)), 'pink':  ('color',(180,100,255)),
    'magenta': ('color',(255,  0,200)), 'brown': ('color',( 40, 80,150)),
    'white':   ('color',(255,255,255)), 'gray':  ('color',(160,160,160)),
    'grey':    ('color',(160,160,160)), 'black': ('color',( 10, 10, 10)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_model():
    if os.path.exists(MODEL_PATH): return
    print(f"\n  Downloading hand model (~3 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  ✓ Done!\n")
    except Exception as e:
        print(f"  [✗] {e}"); raise SystemExit(1)


def get_gesture(lm):
    """
    FIX: More permissive detection.
    Ring/pinky need a larger y-gap to be considered 'up' so accidental
    slight raises don't break the two-finger gesture.
    """
    index_up  = lm[8].y  < lm[6].y  - 0.02
    middle_up = lm[12].y < lm[10].y - 0.02
    ring_up   = lm[16].y < lm[14].y - 0.045   # stricter threshold
    pinky_up  = lm[20].y < lm[18].y - 0.045   # stricter threshold
    pinch_d   = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) * CAM_W

    if pinch_d < 38 and not index_up:
        return 'pinch', pinch_d
    if index_up and middle_up and not ring_up and not pinky_up:
        return 'two_fingers', pinch_d
    if index_up and not middle_up:
        return 'one_finger', pinch_d
    return 'idle', pinch_d


def draw_skeleton(frame, lm, w, h):
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 110, 30), 1)
    for i, p in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, p, r, (0, 185, 255), -1)


def flood_fill(canvas, x, y, color):
    h, w = canvas.shape[:2]
    x, y = max(0, min(w-1, x)), max(0, min(h-1, y))
    mask = np.zeros((h+2, w+2), np.uint8)
    fc   = (int(color[0]), int(color[1]), int(color[2]))
    cv2.floodFill(canvas, mask, (x,y), fc,
                  loDiff=(35,35,35), upDiff=(35,35,35),
                  flags=cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
    canvas[mask[1:-1, 1:-1] == 1] = fc


# ─────────────────────────────────────────────────────────────────────────────
# UI drawing primitives
# ─────────────────────────────────────────────────────────────────────────────
def gradient_fill(img, x1, y1, x2, y2, c1, c2, vertical=True):
    """Fast numpy gradient fill."""
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    if x2 <= x1 or y2 <= y1: return
    if vertical:
        n = y2 - y1
        t = np.linspace(0, 1, n)
        cols = (np.outer(1-t, c1) + np.outer(t, c2)).astype(np.uint8)
        img[y1:y2, x1:x2] = cols[:, np.newaxis, :]
    else:
        n = x2 - x1
        t = np.linspace(0, 1, n)
        cols = (np.outer(1-t, c1) + np.outer(t, c2)).astype(np.uint8)
        img[y1:y2, x1:x2] = cols[np.newaxis, :, :]


def rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a rounded rectangle (filled or outline)."""
    r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
    if thickness == -1:
        cv2.rectangle(img, (x1+r, y1),   (x2-r, y2),   color, -1)
        cv2.rectangle(img, (x1,   y1+r), (x2,   y2-r), color, -1)
        cv2.circle(img, (x1+r, y1+r), r, color, -1)
        cv2.circle(img, (x2-r, y1+r), r, color, -1)
        cv2.circle(img, (x1+r, y2-r), r, color, -1)
        cv2.circle(img, (x2-r, y2-r), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)


def alpha_blend(img, x1, y1, x2, y2, color, alpha=0.40):
    """Semi-transparent solid-color rectangle overlay."""
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    if x2 <= x1 or y2 <= y1: return
    roi     = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0,0), (x2-x1, y2-y1), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, img[y1:y2, x1:x2])


def text_centered(img, txt, cx, cy, scale, color, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    cv2.putText(img, txt, (cx - tw//2, cy + th//2), font, scale, color, thick)


# ─────────────────────────────────────────────────────────────────────────────
# HSV Color Picker
# ─────────────────────────────────────────────────────────────────────────────
class ColorPicker:
    W, H, HUE_H = 280, 230, 28

    def __init__(self):
        self.visible   = False
        self.hue       = 0
        self.sq_h      = self.H - self.HUE_H - 8
        self.hue_strip = self._make_hue_strip()
        self.panel     = None
        self._regen()

    def _make_hue_strip(self):
        hues = np.linspace(0, 179, self.W, dtype=np.uint8)
        hsv  = np.stack([hues, np.full(self.W,255,np.uint8),
                         np.full(self.W,255,np.uint8)], axis=1).reshape(1,self.W,3)
        bgr  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return np.tile(bgr, (self.HUE_H, 1, 1))

    def _regen(self):
        s_row = np.linspace(0, 255, self.W, dtype=np.uint8)
        v_col = np.linspace(255, 0, self.sq_h, dtype=np.uint8)
        s_g   = np.tile(s_row, (self.sq_h, 1))
        v_g   = np.tile(v_col.reshape(-1,1), (1, self.W))
        h_g   = np.full((self.sq_h, self.W), self.hue, dtype=np.uint8)
        sq    = cv2.cvtColor(np.stack([h_g,s_g,v_g], axis=2), cv2.COLOR_HSV2BGR)
        panel = np.full((self.H, self.W, 3), 30, dtype=np.uint8)
        panel[:self.HUE_H]                         = self.hue_strip
        panel[self.HUE_H+4:self.HUE_H+4+self.sq_h] = sq
        self.panel = panel

    def pick_from(self, x, y):
        if not (0 <= x < self.W): return None
        if 0 <= y < self.HUE_H:
            self.hue = int(x*179/self.W); self._regen(); return None
        sy = y - self.HUE_H - 4
        if 0 <= sy < self.sq_h:
            s   = int(x*255/self.W)
            v   = int(255 - sy*255/self.sq_h)
            bgr = cv2.cvtColor(np.array([[[self.hue,s,v]]],np.uint8),
                               cv2.COLOR_HSV2BGR)[0,0]
            return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        return None

    def draw_on(self, frame, ox, oy):
        if self.panel is None: return
        h, w = self.panel.shape[:2]
        fh, fw = frame.shape[:2]
        x2, y2 = min(ox+w, fw), min(oy+h, fh)
        frame[oy:y2, ox:x2] = self.panel[:y2-oy, :x2-ox]
        rounded_rect(frame, ox-2, oy-2, x2+2, y2+2, 6, (190,185,210), 2)
        cv2.putText(frame, 'HSV PICKER', (ox+8, oy+self.HUE_H-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230,228,245), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Emoji Renderer (PIL-backed)
# ─────────────────────────────────────────────────────────────────────────────
class EmojiRenderer:
    SIZE = 56
    FONT_CANDIDATES = [
        "C:/Windows/Fonts/seguiemj.ttf",
        "seguiemj.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    ]

    def __init__(self):
        self.cache = {}; self.font_path = None; self.ok = False
        if PIL_OK: self._find_font()

    def _find_font(self):
        for p in self.FONT_CANDIDATES:
            if os.path.exists(p):
                try: ImageFont.truetype(p, 40); self.font_path=p; self.ok=True; return
                except: pass

    def render(self, char, size=None):
        if not self.ok: return None
        sz  = size or self.SIZE
        key = (char, sz)
        if key in self.cache: return self.cache[key]
        try:
            img  = Image.new('RGBA', (sz,sz), (0,0,0,0))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(self.font_path, sz-6)
            draw.text((3,2), char, font=font, embedded_color=True)
            arr  = np.array(img)
            bgra = arr[:,:,[2,1,0,3]]
            self.cache[key] = bgra; return bgra
        except: return None

    def stamp(self, canvas, char, cx, cy, size=56):
        bgra = self.render(char, size)
        if bgra is None:
            cv2.circle(canvas,(cx,cy),size//2,(100,200,255),-1); return
        h,w   = bgra.shape[:2]
        x1,y1 = cx-w//2, cy-h//2; x2,y2 = x1+w, y1+h
        cx1,cy1 = max(0,x1), max(0,y1)
        cx2,cy2 = min(canvas.shape[1],x2), min(canvas.shape[0],y2)
        if cx2<=cx1 or cy2<=cy1: return
        sl    = bgra[cy1-y1:cy2-y1, cx1-x1:cx2-x1]
        alpha = sl[:,:,3:4]/255.0
        for c in range(3):
            canvas[cy1:cy2,cx1:cx2,c] = (
                alpha[:,:,0]*sl[:,:,c] + (1-alpha[:,:,0])*canvas[cy1:cy2,cx1:cx2,c]
            ).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Voice Controller — FIX: fuzzy matching + stability improvements
# ─────────────────────────────────────────────────────────────────────────────
class VoiceController:
    RECAL_INTERVAL = 90  # seconds between ambient-noise recalibration

    def __init__(self, callback):
        self.cb       = callback
        self.running  = False
        self.last_cmd = ''
        self.last_t   = 0.0
        self.status   = 'off'   # 'off'|'listening'|'processing'|'heard'|'error'
        self._recal_t = 0.0

    # ── FIX: word-level fuzzy matching ──────────────────────────────────────
    @staticmethod
    def _match(text):
        words = text.lower().split()
        # 1. Try multi-word keys longest-first (e.g. "color red")
        for phrase in sorted(VOICE_MAP.keys(), key=len, reverse=True):
            if phrase in text:
                return VOICE_MAP[phrase]
        # 2. Fall back to individual word match
        for word in words:
            if word in VOICE_MAP:
                return VOICE_MAP[word]
        return None

    def start(self):
        if not VOICE_OK:
            print("  [!] Voice disabled — run: pip install SpeechRecognition pyaudio\n")
            self.status = 'off'; return
        self.running = True; self.status = 'listening'
        threading.Thread(target=self._loop, daemon=True).start()
        print("  ✓  Voice active — say: 'red' 'clear' 'pen' 'undo' 'bigger' etc.\n")

    def stop(self): self.running = False; self.status = 'off'

    def _loop(self):
        rec = sr.Recognizer()
        rec.energy_threshold                  = 400
        rec.dynamic_energy_threshold          = True
        rec.dynamic_energy_adjustment_damping = 0.15
        rec.pause_threshold                   = 0.6   # faster phrase end detection

        retry_wait = 1.0
        while self.running:
            try:
                with sr.Microphone() as src:
                    print("  🎤  Calibrating mic for ambient noise…")
                    rec.adjust_for_ambient_noise(src, 1.0)
                    self._recal_t = time.time()
                    retry_wait    = 1.0   # reset backoff on success

                    while self.running:
                        # Periodic recalibration
                        if time.time() - self._recal_t > self.RECAL_INTERVAL:
                            rec.adjust_for_ambient_noise(src, 0.5)
                            self._recal_t = time.time()
                        try:
                            self.status = 'listening'
                            audio  = rec.listen(src, timeout=1.5, phrase_time_limit=5)
                            self.status = 'processing'
                            text   = rec.recognize_google(audio).lower()
                            print(f"  🎤  Heard: \"{text}\"")
                            action = self._match(text)
                            if action:
                                self.last_cmd = text; self.last_t = time.time()
                                self.status   = 'heard'
                                self.cb(action)
                            else:
                                self.status = 'listening'
                        except sr.WaitTimeoutError:
                            self.status = 'listening'
                        except sr.UnknownValueError:
                            self.status = 'listening'
                        except sr.RequestError as e:
                            print(f"  [!] Voice API: {e}")
                            self.status = 'error'; time.sleep(3)

            except OSError as e:
                print(f"  [!] Mic unavailable: {e}"); self.status='off'; break
            except Exception as e:
                print(f"  [!] Voice error: {e}")
                self.status = 'error'
                time.sleep(min(retry_wait, 30)); retry_wait *= 2

    @property
    def recent(self):
        return self.last_cmd if time.time() - self.last_t < 2.5 else ''


# ─────────────────────────────────────────────────────────────────────────────
# Canvas History
# ─────────────────────────────────────────────────────────────────────────────
class History:
    def __init__(self, limit=UNDO_LIMIT):
        self.stack = []; self.future = []; self.limit = limit

    def push(self, canvas):
        self.stack.append(canvas.copy())
        if len(self.stack) > self.limit: self.stack.pop(0)
        self.future.clear()

    def undo(self, canvas):
        if len(self.stack) > 1: self.future.append(self.stack.pop())
        return self.stack[-1].copy()

    def redo(self, canvas):
        if self.future: s=self.future.pop(); self.stack.append(s); return s.copy()
        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Toolbar  — gradient design with rounded swatches
# ─────────────────────────────────────────────────────────────────────────────
class Toolbar:
    R1Y1, R1Y2 = 10, 68      # color row
    R2Y1, R2Y2 = 78, 138     # tool+brush row
    SEP_Y      = 73

    def __init__(self, width, colour_idx=0, brush_idx=2, tool='pen', eraser=False):
        self.width      = width
        self.colour_idx = colour_idx
        self.brush_idx  = brush_idx
        self.tool       = tool
        self.eraser     = eraser
        self.rects      = []
        self.img        = self._render()

    def _render(self):
        img = np.zeros((TB_H, self.width, 3), dtype=np.uint8)

        # Background gradient
        gradient_fill(img, 0, 0, self.width, TB_H, C_BG, C_BG2)
        # Accent strip at top
        gradient_fill(img, 0, 0, self.width, 3, C_ACCENT, (0,140,200))
        # Row separator
        cv2.line(img, (12, self.SEP_Y), (self.width-12, self.SEP_Y), C_SEP, 1)

        self.rects.clear()
        M, SW = 10, 50  # margin, swatch width

        # ── Color swatches ──────────────────────────────────────────────────
        x = M
        for i, (name, col) in enumerate(PALETTE):
            sel = (i == self.colour_idx) and not self.eraser
            if sel:
                # Subtle glow behind selected swatch
                gc = tuple(min(255, c+45) for c in col)
                rounded_rect(img, x-2, self.R1Y1-2, x+SW+2, self.R1Y2+2, 7, gc, -1)
                rounded_rect(img, x, self.R1Y1, x+SW, self.R1Y2, 6, col, -1)
                rounded_rect(img, x, self.R1Y1, x+SW, self.R1Y2, 6, (255,255,255), 2)
                # Arrow indicator
                pts = np.array([[x+SW//2-5,self.R1Y2+2],
                                [x+SW//2+5,self.R1Y2+2],
                                [x+SW//2,  self.R1Y2+8]], np.int32)
                cv2.fillPoly(img, [pts], (255,255,255))
            else:
                rounded_rect(img, x, self.R1Y1, x+SW, self.R1Y2, 6, col, -1)
                rounded_rect(img, x, self.R1Y1, x+SW, self.R1Y2, 6, C_BORDER, 1)
            self.rects.append(('colour', i, x, self.R1Y1, x+SW, self.R1Y2))
            x += SW + 4

        # ── RGB Picker button ───────────────────────────────────────────────
        x += 6
        px1, px2 = x, x+52
        # Rainbow gradient inside picker button
        for dy in range(self.R1Y1+3, self.R1Y2-3):
            t   = (dy - self.R1Y1) / (self.R1Y2 - self.R1Y1)
            hv  = int(t * 179)
            bgr = cv2.cvtColor(np.array([[[hv, 210, 230]]],np.uint8), cv2.COLOR_HSV2BGR)[0,0]
            cv2.line(img, (px1+3,dy), (px2-3,dy), (int(bgr[0]),int(bgr[1]),int(bgr[2])), 1)
        rounded_rect(img, px1, self.R1Y1, px2, self.R1Y2, 6, (0,0,0,0), 1)
        cv2.putText(img,'RGB',(px1+9,(self.R1Y1+self.R1Y2)//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
        self.rects.append(('picker',-1,px1,self.R1Y1,px2,self.R1Y2))

        # ── Tool buttons ────────────────────────────────────────────────────
        tx = M; TW = 68
        for tool in TOOLS:
            is_act = (tool==self.tool and not self.eraser) or (tool=='eraser' and self.eraser)
            bg = C_TOOL_ACT if is_act else C_TOOL_BG
            br = C_ACTIVE   if is_act else C_BORDER
            bw = 2          if is_act else 1
            rounded_rect(img, tx, self.R2Y1, tx+TW, self.R2Y2, 7, bg, -1)
            rounded_rect(img, tx, self.R2Y1, tx+TW, self.R2Y2, 7, br, bw)
            lbl  = TOOL_LABEL[tool]
            tcol = C_ACTIVE if is_act else C_TEXT
            text_centered(img, lbl, tx+TW//2, (self.R2Y1+self.R2Y2)//2, 0.36, tcol, 1)
            self.rects.append(('tool', tool, tx, self.R2Y1, tx+TW, self.R2Y2))
            tx += TW + 4

        # ── Brush size buttons ───────────────────────────────────────────────
        tx += 14
        for i, sz in enumerate(BRUSH_SIZES):
            sel = (i == self.brush_idx)
            bg  = (55,50,70)    if sel else C_TOOL_BG
            br  = C_ACCENT      if sel else C_BORDER
            bw  = 2             if sel else 1
            bx2 = tx + 44
            rounded_rect(img, tx, self.R2Y1, bx2, self.R2Y2, 7, bg, -1)
            rounded_rect(img, tx, self.R2Y1, bx2, self.R2Y2, 7, br, bw)
            dot_col = C_ACCENT if sel else (175,172,195)
            cv2.circle(img, (tx+22, (self.R2Y1+self.R2Y2)//2), min(sz,15), dot_col, -1)
            self.rects.append(('brush', i, tx, self.R2Y1, bx2, self.R2Y2))
            tx += 48

        # ── App title ────────────────────────────────────────────────────────
        cv2.putText(img,'VIRTUAL PAINTER v2',(self.width-226,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,C_MUTED,1)
        return img

    def rebuild(self): self.img = self._render()

    def hit(self, x, y):
        for k,v,x1,y1,x2,y2 in self.rects:
            if x1<=x<=x2 and y1<=y<=y2: return k,v
        return None, None

    def draw_hover(self, frame, x, y):
        k,v = self.hit(x,y)
        if not k: return
        for kk,vv,x1,y1,x2,y2 in self.rects:
            if kk==k and vv==v:
                alpha_blend(frame, x1-1, y1-1, x2+1, y2+1, C_HOVER, 0.22)
                rounded_rect(frame, x1, y1, x2, y2, 6, C_HOVER, 2)
                break


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Painter — Main class
# ─────────────────────────────────────────────────────────────────────────────
class VirtualPainter:
    def __init__(self, W=CAM_W, H=CAM_H):
        self.W, self.H = W, H
        # Drawing state
        self.colour_idx = 0
        self.custom_col = None
        self.brush_size = BRUSH_SIZES[2]
        self.brush_idx  = 2
        self.tool       = 'pen'
        self.eraser     = False
        self.canvas     = np.zeros((H,W,3), np.uint8)
        self.hist       = History(); self.hist.push(self.canvas)
        # Shape
        self.shape_start    = None
        self.shape_preview  = None
        self.last_draw_xy   = None   # FIX: commit shapes at last-drawn point
        # Text
        self.text_mode = False; self.text_pos = None; self.text_buf = ''
        # Emoji
        self.emoji_idx = 0; self.emoji_r = EmojiRenderer()
        # UI
        self.toolbar   = Toolbar(W)
        self.picker    = ColorPicker()
        self.pick_ox   = W//2 - ColorPicker.W//2
        self.pick_oy   = TB_H + 8
        # FIX: separate cooldown timers per action
        self.tb_click_t    = 0.0
        self.shape_click_t = 0.0
        self.emoji_click_t = 0.0
        self.picker_click_t= 0.0
        self.save_flash    = 0.0
        self.confirm_flash = 0.0    # toolbar selection flash
        self.cmd_msg       = ''; self.cmd_t = 0.0
        # Gesture state machine
        self.g_buf     = deque(maxlen=STABLE_FRAMES)
        self.g_cur     = 'idle'
        self.prev_xy   = None
        self.smooth_xy = (0, 0)
        self.raw_prev  = (0, 0)
        self.fill_done = False
        # MediaPipe
        ensure_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.70)
        self.lmk     = mp_vision.HandLandmarker.create_from_options(opts)
        self.ts_base = int(time.time()*1000); self.last_ts = 0
        # Voice
        self.voice = VoiceController(self._voice_action); self.voice.start()
        # FPS (smoothed)
        self._fps_t   = time.time()
        self._fps_buf = deque([30.0]*10, maxlen=10)

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def active_color(self):
        if self.eraser: return (0,0,0)
        return self.custom_col or PALETTE[self.colour_idx][1]

    # ── Voice callback ──────────────────────────────────────────────────────
    def _voice_action(self, action):
        if   action == 'clear':
            self._push(); self.canvas[:]=0; self._msg("🎤 Canvas cleared")
        elif action == 'save':
            self._save(); self._msg("🎤 Saved!")
        elif action == 'undo':
            self._undo(); self._msg("🎤 Undo")
        elif action == 'redo':
            self._redo(); self._msg("🎤 Redo")
        elif action == 'brush_up':
            self.brush_idx  = min(len(BRUSH_SIZES)-1, self.brush_idx+1)
            self.brush_size = BRUSH_SIZES[self.brush_idx]
            self._msg(f"🎤 Brush: {self.brush_size}px"); self._sync()
        elif action == 'brush_down':
            self.brush_idx  = max(0, self.brush_idx-1)
            self.brush_size = BRUSH_SIZES[self.brush_idx]
            self._msg(f"🎤 Brush: {self.brush_size}px"); self._sync()
        elif isinstance(action, tuple):
            k, v = action
            if k == 'tool':
                self.eraser = (v=='eraser'); self.tool = 'pen' if v=='eraser' else v
                self._msg(f"🎤 {v.title()}"); self._sync()
            elif k == 'color':
                self.custom_col=v; self.eraser=False
                self._msg("🎤 Color changed"); self._sync()

    def _msg(self, t):  self.cmd_msg=t; self.cmd_t=time.time()
    def _push(self):    self.hist.push(self.canvas)
    def _undo(self):    self.canvas=self.hist.undo(self.canvas)
    def _redo(self):    self.canvas=self.hist.redo(self.canvas)
    def _save(self):
        os.makedirs('paintings', exist_ok=True)
        p = f'paintings/painting_{datetime.now():%Y%m%d_%H%M%S}.png'
        cv2.imwrite(p, self.canvas)
        self.save_flash = time.time() + 2.5
        print(f'  ✓  Saved → {os.path.abspath(p)}')

    def _sync(self):
        self.toolbar.colour_idx=self.colour_idx; self.toolbar.brush_idx=self.brush_idx
        self.toolbar.tool=self.tool; self.toolbar.eraser=self.eraser
        self.toolbar.rebuild()

    def _tb_action(self, kind, val):
        if   kind=='colour': self.colour_idx=val; self.custom_col=None; self.eraser=False
        elif kind=='brush':  self.brush_idx=val;  self.brush_size=BRUSH_SIZES[val]
        elif kind=='tool':
            if val=='eraser': self.eraser=True;  self.tool='pen'
            else:             self.eraser=False; self.tool=val
        elif kind=='picker': self.picker.visible = not self.picker.visible
        self._sync()
        self.tb_click_t    = time.time()      # FIX: only toolbar cooldown reset
        self.confirm_flash = time.time()+0.4  # visual confirmation flash

    def _draw_shape(self, cv, p1, p2, col, thick):
        if   self.tool=='line':   cv2.line(cv,p1,p2,col,thick)
        elif self.tool=='rect':   cv2.rectangle(cv,p1,p2,col,thick)
        elif self.tool=='circle':
            r=int(math.hypot(p2[0]-p1[0],p2[1]-p1[1]))
            cv2.circle(cv,p1,r,col,thick)

    # ── Main frame processing ───────────────────────────────────────────────
    def process(self, frame):
        frame = cv2.flip(frame, 1)
        if frame.shape[:2] != (self.H, self.W):
            frame = cv2.resize(frame, (self.W, self.H))

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts    = max(int(time.time()*1000)-self.ts_base, self.last_ts+1)
        self.last_ts = ts
        res   = self.lmk.detect_for_video(mp_im, ts)

        mode = 'idle'; sx,sy = self.smooth_xy

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            draw_skeleton(frame, lm, self.W, self.H)

            rx,ry = int(lm[8].x*self.W), int(lm[8].y*self.H)
            prx,pry = self.raw_prev
            speed   = math.hypot(rx-prx, ry-pry)
            alpha   = min(0.88, max(0.30, speed/45.0))
            self.raw_prev  = (rx,ry)
            sx = int(alpha*rx+(1-alpha)*sx)
            sy = int(alpha*ry+(1-alpha)*sy)
            self.smooth_xy = (sx,sy)

            # ── Gesture hysteresis ────────────────────────────────────────
            raw_g, pinch_d = get_gesture(lm)
            self.g_buf.append(raw_g)
            if len(self.g_buf)==STABLE_FRAMES and len(set(self.g_buf))==1:
                self.g_cur = raw_g
            g = self.g_cur

            # ── PINCH → resize brush ──────────────────────────────────────
            if g=='pinch':
                mode='pinch'
                self.brush_size = int(np.interp(pinch_d,[15,120],[MIN_BRUSH,MAX_BRUSH]))
                r = self.brush_size
                cv2.circle(frame,(sx,sy),r,C_ACCENT,2)
                cv2.circle(frame,(sx,sy),r//2,(255,220,100),1)
                cv2.circle(frame,(sx,sy),3,(255,255,255),-1)
                cv2.putText(frame,f'{r}px',(sx+r+10,sy),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,C_ACCENT,2)
                self.prev_xy = None

            # ── TWO FINGERS → SELECT / confirm ────────────────────────────
            elif g=='two_fingers':
                mode='select'; self.prev_xy=None; self.fill_done=False
                now = time.time()

                # FIX: commit shape at LAST DRAW position (not two-finger pos)
                if (self.tool in ('line','rect','circle')
                        and self.shape_start is not None
                        and (now - self.shape_click_t) > SHAPE_COOLDOWN):
                    end_pt = self.last_draw_xy or (sx,sy)
                    if end_pt[1] > TB_H:
                        self._push()
                        self._draw_shape(self.canvas, self.shape_start, end_pt,
                                         self.active_color, self.brush_size)
                    self.shape_start   = None
                    self.shape_preview = None
                    self.shape_click_t = now

                # Emoji stamp
                if (self.tool=='emoji' and sy>TB_H
                        and (now-self.emoji_click_t)>EMOJI_COOLDOWN):
                    self._push()
                    self.emoji_r.stamp(self.canvas,EMOJIS[self.emoji_idx],
                                       sx,sy,max(40,self.brush_size*3))
                    self.emoji_click_t = now

                # FIX: toolbar uses its OWN cooldown (tb_click_t)
                if sy<TB_H and (now-self.tb_click_t)>TB_COOLDOWN:
                    k,v = self.toolbar.hit(sx,sy)
                    if k: self._tb_action(k,v)

                # Color picker
                if (self.picker.visible
                        and (now-self.picker_click_t)>PICKER_COOLDOWN):
                    lx,ly = sx-self.pick_ox, sy-self.pick_oy
                    if 0<=lx<ColorPicker.W and 0<=ly<ColorPicker.H:
                        c = self.picker.pick_from(lx,ly)
                        if c: self.custom_col=c; self.picker_click_t=now

                # Two-finger cursor
                cv2.circle(frame,(sx,sy),20,(255,255,255),2)
                cv2.circle(frame,(sx,sy),7,self.active_color,-1)
                cv2.circle(frame,(sx,sy),7,(255,255,255),1)

            # ── ONE FINGER → DRAW ─────────────────────────────────────────
            elif g=='one_finger':
                mode='draw'
                if sy>TB_H:
                    col=self.active_color; sz=self.brush_size

                    if self.eraser or self.tool=='pen':
                        dc = (0,0,0) if self.eraser else col
                        ds = sz*3    if self.eraser else sz
                        if self.prev_xy:
                            cv2.line(self.canvas,self.prev_xy,(sx,sy),dc,ds*2)
                            cv2.circle(self.canvas,(sx,sy),ds,dc,-1)
                        self.prev_xy=self.last_draw_xy=(sx,sy)

                    elif self.tool=='fill':
                        if not self.fill_done:
                            self._push(); flood_fill(self.canvas,sx,sy,col)
                            self.fill_done=True
                        self.prev_xy=(sx,sy)

                    elif self.tool=='text':
                        if self.prev_xy is None:
                            self.text_mode=True; self.text_pos=(sx,sy); self.text_buf=''
                        self.prev_xy=(sx,sy)

                    elif self.tool=='emoji':
                        self.prev_xy=(sx,sy)

                    elif self.tool in ('line','rect','circle'):
                        if self.shape_start is None: self.shape_start=(sx,sy)
                        prev=self.canvas.copy()
                        self._draw_shape(prev,self.shape_start,(sx,sy),col,sz)
                        self.shape_preview = prev
                        self.prev_xy = self.last_draw_xy = (sx,sy)  # FIX: track end
                else:
                    self.prev_xy=None

               # Cursor rendering
                cursor_size = self.brush_size

                if self.eraser:
                     cv2.circle(frame, (sx, sy), cursor_size * 3, (140,140,255), 2)
                     cv2.circle(frame, (sx, sy), 3, (255,255,255), -1)
                else:
                    cv2.circle(frame, (sx, sy), max(cursor_size, 6), self.active_color, 2)
                    cv2.circle(frame, (sx, sy), 3, (255,255,255), -1)

            else:  # idle
                self.prev_xy=None; self.fill_done=False
                self.last_draw_xy=None; mode='idle'

        # ── Composite canvas onto frame ────────────────────────────────────
        rc    = self.shape_preview if self.shape_preview is not None else self.canvas
        gray  = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY)
        _,mask= cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        out   = cv2.add(cv2.bitwise_and(frame,frame,mask=cv2.bitwise_not(mask)),
                        cv2.bitwise_and(rc,rc,mask=mask))
        self.shape_preview = None

        # ── Toolbar ────────────────────────────────────────────────────────
        out[:TB_H,:] = self.toolbar.img
        if mode=='select' and sy<TB_H:
            self.toolbar.draw_hover(out,sx,sy)

        # ── Confirmation flash (green tint over toolbar) ───────────────────
        if time.time() < self.confirm_flash:
            t = (self.confirm_flash - time.time()) / 0.4
            alpha_blend(out,0,0,self.W,TB_H, C_ACTIVE, min(0.55, t*0.55))

        # ── Overlay panels ─────────────────────────────────────────────────
        if self.picker.visible:
            self.picker.draw_on(out, self.pick_ox, self.pick_oy)
        if self.tool=='emoji' and not self.eraser:
            self._draw_emoji_panel(out)
        if self.text_mode and self.text_pos:
            self._draw_text_cursor(out)

        # ── Bottom status bar ──────────────────────────────────────────────
        self._draw_status_bar(out, mode)

        # ── Voice command message ──────────────────────────────────────────
        if time.time()-self.cmd_t < 2.5 and self.cmd_msg:
            y_msg = self.H - SB_H - 14
            (tw,th),_ = cv2.getTextSize(self.cmd_msg,cv2.FONT_HERSHEY_SIMPLEX,0.60,2)
            alpha_blend(out,10,y_msg-th-6,16+tw,y_msg+6,(0,0,0),0.60)
            cv2.putText(out,self.cmd_msg,(12,y_msg),
                        cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,218,50),2)

        # ── Save flash ─────────────────────────────────────────────────────
        if time.time() < self.save_flash:
            t = min(1.0,(self.save_flash-time.time())/0.5)
            alpha_blend(out,self.W//2-190,self.H//2-60,
                            self.W//2+190,self.H//2+60,(0,0,0),0.70*t)
            cv2.putText(out,'✓  SAVED!',(self.W//2-118,self.H//2+24),
                        cv2.FONT_HERSHEY_DUPLEX,1.7,C_ACTIVE,3)

        # ── Voice badge ────────────────────────────────────────────────────
        self._draw_voice_badge(out)

        return out

    # ── Status bar ─────────────────────────────────────────────────────────
    def _draw_status_bar(self, frame, mode):
        y1,y2 = self.H-SB_H, self.H
        gradient_fill(frame,0,y1,self.W,y2,C_SB,(10,9,15))
        cv2.line(frame,(0,y1),(self.W,y1),C_SEP,1)

        now = time.time()
        self._fps_buf.append(1.0/max(now-self._fps_t,1e-9))
        self._fps_t = now
        fps = sum(self._fps_buf)/len(self._fps_buf)

        cy = (y1+y2)//2 + 6  # text baseline

        # Mode
        mode_info = {
            'idle':   ('● IDLE',   C_MUTED),
            'select': ('✌ SELECT', C_SELECT),
            'pinch':  (f'⊙ BRUSH {self.brush_size}px', C_ACCENT),
            'draw':   (('✏ ERASING' if self.eraser else f'✏ {self.tool.upper()}'), C_ACTIVE),
        }
        lbl, col = mode_info[mode]
        cv2.putText(frame,lbl,(14,cy),cv2.FONT_HERSHEY_SIMPLEX,0.52,col,1)

        # Color preview
        if not self.eraser:
            cx = self.W//2 - 55
            cv2.circle(frame,(cx,cy-5),11,self.active_color,-1)
            cv2.circle(frame,(cx,cy-5),11,C_BORDER,1)
            name = 'CUSTOM' if self.custom_col else PALETTE[self.colour_idx][0].upper()
            cv2.putText(frame,name,(cx+17,cy),cv2.FONT_HERSHEY_SIMPLEX,0.40,C_TEXT,1)
        else:
            cv2.putText(frame,'ERASER',(self.W//2-30,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(150,148,200),1)

        # Brush size
        bx = self.W//2 + 60
        cv2.circle(frame,(bx,cy-5),min(self.brush_size,13),C_TEXT,-1)
        cv2.putText(frame,f'{self.brush_size}px',(bx+17,cy),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,C_MUTED,1)

        # Shortcut hint
        cv2.putText(frame,'Z=Undo  Y=Redo  S=Save  C=Clear  T=Text',
                    (self.W//2+100,cy),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_MUTED,1)

        # FPS
        cv2.putText(frame,f'FPS {fps:.0f}',(self.W-80,cy),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,C_MUTED,1)

    # ── Voice status badge ──────────────────────────────────────────────────
    def _draw_voice_badge(self, frame):
        st = self.voice.status
        bx,by = self.W-86, 6; bw,bh = 80, 26

        cfg = {
            'off':        ((55,50,68),   C_MUTED),
            'listening':  ((25,55,25),   C_ACTIVE),
            'processing': ((50,75,18),   C_ACCENT),
            'heard':      ((18,65,18),   C_ACTIVE),
            'error':      ((55,25,55),   (130,60,200)),
        }
        bg, fc = cfg.get(st, cfg['off'])
        rounded_rect(frame,bx,by,bx+bw,by+bh,5,bg,-1)
        rounded_rect(frame,bx,by,bx+bw,by+bh,5,fc,1)

        # Animated dot
        if st=='listening':
            pulse = 0.5+0.5*math.sin(time.time()*5)
            cv2.circle(frame,(bx+12,by+13),int(3+2*pulse),fc,-1)
        else:
            cv2.circle(frame,(bx+12,by+13),4,fc,-1 if st=='heard' else 1)

        label={'off':'OFF','listening':'MIC','processing':'...','heard':'HEARD','error':'ERR'}
        cv2.putText(frame,label.get(st,'MIC'),(bx+22,by+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,fc,1)

        rc = self.voice.recent
        if rc:
            short = rc[:10]+'..' if len(rc)>10 else rc
            cv2.putText(frame,f'"{short}"',(bx-95,by+18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.33,(195,190,150),1)

    # ── Emoji panel ─────────────────────────────────────────────────────────
    def _draw_emoji_panel(self, frame):
        px,py=8,TB_H+8; ew=58
        for i,e in enumerate(EMOJIS):
            y=py+i*(ew+4); sel=(i==self.emoji_idx)
            rounded_rect(frame,px,y,px+ew,y+ew,8,(65,60,80) if sel else (32,30,44),-1)
            rounded_rect(frame,px,y,px+ew,y+ew,8,C_ACTIVE if sel else C_BORDER,2 if sel else 1)
            bgra=self.emoji_r.render(e,48)
            if bgra is not None:
                ry1,ry2=y+5,y+53; rx1,rx2=px+5,px+53
                sl=bgra[:ry2-ry1,:rx2-rx1]; a=sl[:,:,3:4]/255.0
                for c in range(3):
                    frame[ry1:ry2,rx1:rx2,c]=(
                        a[:,:,0]*sl[:,:,c]+(1-a[:,:,0])*frame[ry1:ry2,rx1:rx2,c]
                    ).astype(np.uint8)
            else:
                text_centered(frame,str(i+1),px+ew//2,y+ew//2,0.8,C_TEXT,2)

    # ── Text cursor ─────────────────────────────────────────────────────────
    def _draw_text_cursor(self, frame):
        if not self.text_pos: return
        tx,ty = self.text_pos
        cv2.putText(frame,self.text_buf+'|',(tx+1,ty+1),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3)
        cv2.putText(frame,self.text_buf+'|',(tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,self.active_color,2)

    def commit_text(self):
        if self.text_pos and self.text_buf:
            self._push()
            cv2.putText(self.canvas,self.text_buf,self.text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,self.active_color,2)
        self.text_mode=False; self.text_pos=None; self.text_buf=''


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print('\n[✗] Cannot open webcam (index 0). Try VideoCapture(1).\n'); return

    p = VirtualPainter()
    print('\n' + '═'*62)
    print('   🎨  Virtual Painter Pro v2.0')
    print('═'*62)
    print('   ☝  1 finger   → DRAW with selected tool')
    print('   ✌  2 fingers  → SELECT toolbar / confirm shape')
    print('   🤏 Pinch       → Resize brush live')
    print()
    print('   Z=Undo  Y=Redo  S=Save  C=Clear  T=Text mode')
    print('   1-8=Emoji select    Q/ESC=Quit')
    print()
    print('   🎤 Voice: "red" "clear" "pen" "circle" "undo"')
    print('            "bigger" "save" "eraser" "redo" …')
    print('═'*62 + '\n')

    while True:
        ok,frame = cap.read()
        if not ok: break
        cv2.imshow('Virtual Painter Pro v2.0  |  Z=Undo  S=Save  Q=Quit',
                   p.process(frame))
        key = cv2.waitKey(1) & 0xFF
        if   key in (ord('q'),27):                   break
        elif key==ord('s'):                           p._save()
        elif key==ord('c'):                           p._push(); p.canvas[:]=0
        elif key==ord('z'):                           p._undo()
        elif key==ord('y'):                           p._redo()
        elif key==13 and p.text_mode:                p.commit_text()
        elif key==27 and p.text_mode:                p.text_mode=False; p.text_buf=''
        elif p.text_mode and 32<=key<=126:            p.text_buf+=chr(key)
        elif key==8  and p.text_mode:                p.text_buf=p.text_buf[:-1]
        elif ord('1')<=key<=ord('8'):                p.emoji_idx=key-ord('1')

    cap.release(); cv2.destroyAllWindows(); p.voice.stop()
    print('\n  Goodbye! 👋\n')


if __name__ == '__main__':
    main()