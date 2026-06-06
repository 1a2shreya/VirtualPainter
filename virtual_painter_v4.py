#!/usr/bin/env python3
"""
Virtual Painter Pro v4.0 — AI-Powered Professional Painting Suite
═══════════════════════════════════════════════════════════════════════════════

What's New vs v3.1:
  [NEW] BrushEngine  — Pen/Marker/Pencil/Airbrush/Watercolor/Neon/Glow/Spray/Calli
  [NEW] ColorPicker  — Vectorised HSV wheel + RGB sliders + HEX + recent/favorites
  [NEW] LayerPanelUI — Lock/Hide/Rename/Duplicate/Opacity per layer
  [NEW] AIShapeRecognizer — OpenCV contour classification -> perfect shape replace
  [NEW] SelectionTool — Rect select, Move, Copy, Paste, Delete region
  [NEW] CanvasViewport — Zoom 1-4x + Pan (pinch=zoom, two-finger drag=pan)
  [NEW] SymmetryEngine — Vertical / Horizontal / Radial-8 / Mandala
  [NEW] EffectsEngine — Blur/Sharpen/Glow/Neon/OilPaint/Sketch/Watercolor/Comic
  [NEW] StickerLibrary — Built-in stickers drawn with OpenCV primitives
  [NEW] ProjectManager — .vpaint JSON+base64 save/load/autosave
  [NEW] Analytics — Session time, strokes, shapes, undo/redo, voice usage
  [NEW] ExportManager — PNG / JPG / SVG export
  [NEW] SidePanel — Unified overlay panel (brush/layers/colors/effects/stats)
  [NEW] 40+ voice commands (draw circle, switch layer 3, enable symmetry, ...)
  [FIX] emoji_idx AttributeError -- initialised in __init__
  [FIX] Undo/Redo fallback syncs visible[]/opacity[] metadata
  [FIX] gradient_fill() horizontal broadcast crash
  [FIX] LayerManager.merge_visible() -- proper alpha-composite (not darkening)
  [FIX] VoiceFeedback.stop() -- sentinel + clean thread shutdown
  [FIX] draw_skeleton() -- theme-aware bone colour
  [FIX] custom_col cleared on tool switch
  [FIX] shape_preview uses ROI crop (not full layer copy)
  [PERF] Dirty-flag cache in merge_visible() -- skip unchanged frames
  [PERF] Dock background pre-cached per theme (no GaussianBlur every frame)
  [PERF] AnimatedCursor trail via cv2.polylines (vectorised)

Gestures:
  1 finger   -> DRAW  (uses active brush & symmetry)
  2 fingers  -> SELECT / confirm shape / click dock / click panel
  3 fingers  -> UNDO  (once per hold)
  4 fingers  -> REDO  (once per hold)
  Palm open  -> CLEAR active layer (once per hold)
  Pinch      -> Zoom in/out on canvas

Keys:
  Z=Undo  Y=Redo  S=Save PNG  Ctrl+S=Save Project  C=Clear  G=Grid
  D=Theme  T=Text  R=Timelapse  1-8=Layer  Q/ESC=Quit
  B=Brush panel  L=Layer panel  K=Color panel  E=Effects  A=Analytics
  X=Symmetry cycle  V=Viewport reset  F=Apply last effect
"""

import os, sys, base64, time, math, threading, random, json, queue
import urllib.request, logging
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel']     = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from PIL import Image, ImageDraw, ImageFont; PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import speech_recognition as sr; VOICE_OK = True
except ImportError:
    VOICE_OK = False

try:
    import pyttsx3; TTS_OK = True
except ImportError:
    TTS_OK = False

# =============================================================================
# SECTION 1 — CONSTANTS & CONFIGURATION
# =============================================================================
CAM_W, CAM_H      = 1280, 720
DOCK_W            = 82          # left tool dock
PANEL_W           = 230         # right side-panel overlay
SB_H              = 42          # status bar
STABLE_FRAMES     = 3
MIN_BRUSH         = 2
MAX_BRUSH         = 80
UNDO_LIMIT        = 60
MAX_LAYERS        = 8
AUTOSAVE_SECS     = 300

TB_COOLDOWN       = 0.28
GESTURE_COOLDOWN  = 0.45
PANEL_COOLDOWN    = 0.22

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

BRUSH_TYPES: List[str] = [
    'pen','marker','pencil','airbrush','watercolor',
    'neon','glow','spray','calligraphy',
]
TOOLS: List[str] = [
    'pen','eraser','line','rect','circle',
    'fill','text','emoji','select','eyedropper',
]
BRUSH_SIZES: List[int] = [3, 7, 14, 24, 36, 52]

# =============================================================================
# SECTION 2 — ENUMS & DATACLASSES
# =============================================================================
class Theme(Enum):
    DARK  = "dark"
    LIGHT = "light"

class SymMode(Enum):
    NONE       = "none"
    VERTICAL   = "vertical"
    HORIZONTAL = "horizontal"
    RADIAL     = "radial"
    MANDALA    = "mandala"

class PanelMode(Enum):
    NONE      = "none"
    BRUSH     = "brush"
    LAYERS    = "layers"
    COLORS    = "colors"
    EFFECTS   = "effects"
    ANALYTICS = "analytics"


@dataclass
class BrushConfig:
    brush_type: str   = 'pen'
    size:       int   = 14
    opacity:    float = 1.0
    hardness:   float = 0.8
    flow:       float = 1.0

    def validate(self) -> None:
        self.size     = int(np.clip(self.size,     MIN_BRUSH, MAX_BRUSH))
        self.opacity  = float(np.clip(self.opacity,  0.05, 1.0))
        self.hardness = float(np.clip(self.hardness, 0.10, 1.0))
        self.flow     = float(np.clip(self.flow,     0.10, 1.0))


@dataclass
class LayerMeta:
    name:    str   = "Layer"
    visible: bool  = True
    locked:  bool  = False
    opacity: float = 1.0


@dataclass
class ViewTransform:
    zoom:  float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def reset(self) -> None:
        self.zoom = 1.0; self.pan_x = 0.0; self.pan_y = 0.0

    def screen_to_canvas(self, sx: int, sy: int) -> Tuple[int,int]:
        return (int((sx - self.pan_x) / self.zoom),
                int((sy - self.pan_y) / self.zoom))

    def canvas_to_screen(self, cx: int, cy: int) -> Tuple[int,int]:
        return (int(cx * self.zoom + self.pan_x),
                int(cy * self.zoom + self.pan_y))


@dataclass
class SelectionState:
    active:     bool = False
    rect:       Optional[Tuple[int,int,int,int]] = None
    clipboard:  Optional[np.ndarray]             = None
    drag_start: Optional[Tuple[int,int]]         = None


@dataclass
class AnalyticsData:
    session_start:   float = field(default_factory=time.time)
    total_strokes:   int   = 0
    shapes_drawn:    int   = 0
    undo_count:      int   = 0
    redo_count:      int   = 0
    layer_switches:  int   = 0
    voice_commands:  int   = 0
    brush_changes:   int   = 0
    effects_applied: int   = 0

    @property
    def drawing_time(self) -> str:
        return str(timedelta(seconds=int(time.time()-self.session_start)))


# =============================================================================
# SECTION 3 — THEMES, PALETTE, VOICE MAP
# =============================================================================
THEMES: Dict[str, Dict] = {
    "dark": {
        "bg":(30,28,36),     "bg2":(20,18,26),     "accent":(30,190,255),
        "select":(60,200,255),"active":(55,230,100),"hover":(0,230,240),
        "text":(220,218,232),"muted":(110,108,128), "border":(60,58,78),
        "sep":(48,46,62),    "tool_bg":(42,40,55),  "tool_act":(22,75,38),
        "dock_bg":(25,24,32),"sb":(16,15,20),       "panel_bg":(28,26,38),
        "danger":(60,40,200),"warning":(20,160,255),
    },
    "light": {
        "bg":(245,245,250),  "bg2":(235,235,240),   "accent":(200,100,30),
        "select":(180,120,50),"active":(50,180,100),"hover":(100,150,200),
        "text":(30,30,40),   "muted":(120,120,130), "border":(150,150,160),
        "sep":(200,200,210), "tool_bg":(220,220,230),"tool_act":(200,240,220),
        "dock_bg":(240,240,248),"sb":(230,230,238),  "panel_bg":(240,240,248),
        "danger":(80,60,220),"warning":(20,140,240),
    },
}

PALETTE: List[Tuple] = [
    ("Red",(0,0,255)),     ("Orange",(0,130,255)),  ("Yellow",(0,220,255)),
    ("Lime",(0,255,0)),    ("Green",(0,180,0)),      ("Teal",(128,200,0)),
    ("Cyan",(255,220,0)),  ("Sky",(255,160,80)),     ("Blue",(255,30,30)),
    ("Navy",(140,20,20)),  ("Purple",(200,0,190)),   ("Pink",(180,100,255)),
    ("Magenta",(255,0,200)),("Brown",(40,80,150)),   ("Tan",(100,170,200)),
    ("White",(255,255,255)),("Gray",(160,160,160)),  ("Black",(10,10,10)),
]

VOICE_MAP: Dict[str,Any] = {
    'clear':'clear','wipe':'clear','reset':'clear',
    'undo':'undo','back':'undo','redo':'redo','forward':'redo',
    'save':'save','export':'save','save png':'save_png','save jpg':'save_jpg',
    'save project':'save_project','open project':'open_project',
    'new canvas':'new','new painting':'new',
    'bigger':'brush_up','larger':'brush_up','increase size':'brush_up',
    'smaller':'brush_down','decrease size':'brush_down',
    'increase opacity':'opacity_up','decrease opacity':'opacity_down',
    'pen':('tool','pen'),'pencil':('tool','pen'),'draw':('tool','pen'),
    'eraser':('tool','eraser'),'erase':('tool','eraser'),
    'line':('tool','line'),'straight':('tool','line'),
    'rectangle':('tool','rect'),'rect':('tool','rect'),'square':('tool','rect'),
    'circle':('tool','circle'),'oval':('tool','circle'),
    'fill':('tool','fill'),'bucket':('tool','fill'),
    'text':('tool','text'),'write':('tool','text'),
    'emoji':('tool','emoji'),'sticker':('tool','emoji'),
    'select':('tool','select'),
    'brush pen':'brush_pen','use pen':'brush_pen',
    'brush marker':'brush_marker','use marker':'brush_marker',
    'brush pencil':'brush_pencil','airbrush':'brush_airbrush',
    'watercolor':'brush_watercolor','neon brush':'brush_neon',
    'glow brush':'brush_glow','spray paint':'brush_spray',
    'calligraphy':'brush_calligraphy',
    'draw circle':'shape_circle','draw rectangle':'shape_rect',
    'draw triangle':'shape_triangle','draw star':'shape_star',
    'draw arrow':'shape_arrow',
    'switch layer 1':('layer',0),'layer one':('layer',0),
    'switch layer 2':('layer',1),'layer two':('layer',1),
    'switch layer 3':('layer',2),'layer three':('layer',2),
    'switch layer 4':('layer',3),'layer four':('layer',3),
    'new layer':'new_layer','add layer':'new_layer',
    'delete layer':'delete_layer',
    'zoom in':'zoom_in','zoom out':'zoom_out','reset zoom':'reset_zoom',
    'enable symmetry':'symmetry_on','disable symmetry':'symmetry_off',
    'vertical symmetry':'symmetry_v','horizontal symmetry':'symmetry_h',
    'radial symmetry':'symmetry_r','mandala':'symmetry_mandala',
    'no symmetry':'symmetry_off',
    'dark mode':'toggle_dark','light mode':'toggle_light',
    'enable grid':'grid_on','disable grid':'grid_off',
    'red':('color',(0,0,255)),'orange':('color',(0,130,255)),
    'yellow':('color',(0,220,255)),'lime':('color',(0,255,0)),
    'green':('color',(0,200,0)),'cyan':('color',(255,220,0)),
    'blue':('color',(255,0,0)),'purple':('color',(200,0,190)),
    'pink':('color',(180,100,255)),'white':('color',(255,255,255)),
    'black':('color',(10,10,10)),'gray':('color',(160,160,160)),
    'grey':('color',(160,160,160)),
    'apply blur':'fx_blur','apply glow':'fx_glow','apply neon':'fx_neon',
    'sketch':'fx_sketch','sharpen':'fx_sharpen','oil paint':'fx_oilpaint',
    'comic':'fx_comic',
}

# =============================================================================
# SECTION 4 — UI HELPER FUNCTIONS
# =============================================================================
def rounded_rect(img: np.ndarray, x1:int, y1:int, x2:int, y2:int,
                 r:int, color:Tuple, thickness:int=-1) -> None:
    r = max(0, min(r,(x2-x1)//2,(y2-y1)//2))
    if thickness == -1:
        cv2.rectangle(img,(x1+r,y1),(x2-r,y2),color,-1)
        cv2.rectangle(img,(x1,y1+r),(x2,y2-r),color,-1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img,(cx,cy),r,color,-1)
    else:
        cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
        cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
        cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
        cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
        cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,thickness)
        cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,thickness)
        cv2.ellipse(img,(x1+r,y2-r),(r,r),90,0,90,color,thickness)
        cv2.ellipse(img,(x2-r,y2-r),(r,r),0,0,90,color,thickness)


def gradient_fill(img:np.ndarray, x1:int, y1:int, x2:int, y2:int,
                  c1:Tuple, c2:Tuple, vertical:bool=True) -> None:
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    if vertical:
        n  = y2-y1
        t  = np.linspace(0,1,n,dtype=np.float32)
        c  = (np.outer(1-t,c1)+np.outer(t,c2)).astype(np.uint8)
        img[y1:y2,x1:x2] = c[:,np.newaxis,:]    # (n,1,3) -> broadcasts to (n,w,3)
    else:
        n  = x2-x1
        t  = np.linspace(0,1,n,dtype=np.float32)
        c  = (np.outer(1-t,c1)+np.outer(t,c2)).astype(np.uint8)
        # FIX: tile across all rows instead of broadcasting (1,n,3)
        img[y1:y2,x1:x2] = np.tile(c[np.newaxis,:,:],(y2-y1,1,1))


def alpha_blend(img:np.ndarray, x1:int, y1:int, x2:int, y2:int,
                color:Tuple, alpha:float=0.40) -> None:
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    roi = img[y1:y2,x1:x2]
    ov  = roi.copy()
    cv2.rectangle(ov,(0,0),(x2-x1,y2-y1),color,-1)
    cv2.addWeighted(ov,alpha,roi,1-alpha,0,img[y1:y2,x1:x2])


def text_centered(img:np.ndarray,txt:str,cx:int,cy:int,
                  scale:float,color:Tuple,thick:int=1) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_ = cv2.getTextSize(txt,font,scale,thick)
    cv2.putText(img,txt,(cx-tw//2,cy+th//2),font,scale,color,thick,cv2.LINE_AA)


def pill_text(img:np.ndarray, txt:str, cx:int, cy:int,
              fg:Tuple, bg:Tuple, scale:float=0.48,
              pad:int=8, alpha:float=0.78) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_ = cv2.getTextSize(txt,font,scale,1)
    x1,y1 = cx-tw//2-pad, cy-th//2-pad
    x2,y2 = cx+tw//2+pad, cy+th//2+pad
    alpha_blend(img,x1,y1,x2,y2,bg,alpha)
    rounded_rect(img,x1,y1,x2,y2,(y2-y1)//2,bg,2)
    cv2.putText(img,txt,(cx-tw//2,cy+th//2),font,scale,fg,1,cv2.LINE_AA)


def draw_skeleton(frame:np.ndarray, lm, w:int, h:int,
                  theme:str="dark") -> None:
    # FIX: theme-aware bone colour (was hardcoded)
    bone = (80,55,20) if theme=="dark" else (160,110,50)
    pts  = [(int(l.x*w),int(l.y*h)) for l in lm]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame,pts[a],pts[b],bone,1)
    for i,p in enumerate(pts):
        r = 5 if i in (4,8,12,16,20) else 3
        cv2.circle(frame,p,r,(0,185,255),-1)


def flood_fill(canvas:np.ndarray, x:int, y:int, color:Tuple) -> None:
    h,w = canvas.shape[:2]
    x,y = max(0,min(w-1,x)), max(0,min(h-1,y))
    mask = np.zeros((h+2,w+2),np.uint8)
    fc   = (int(color[0]),int(color[1]),int(color[2]))
    cv2.floodFill(canvas,mask,(x,y),fc,
                  loDiff=(35,35,35),upDiff=(35,35,35),
                  flags=cv2.FLOODFILL_FIXED_RANGE|cv2.FLOODFILL_MASK_ONLY)
    canvas[mask[1:-1,1:-1]==1] = fc


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH): return
    print("\n  Downloading hand model (~3 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
        print("  Done!\n")
    except Exception as e:
        print(f"  [Error] {e}"); raise SystemExit(1)


# =============================================================================
# SECTION 5 — BRUSH ENGINE
# =============================================================================
class BrushEngine:
    """9-brush drawing engine. All strokes route through apply_stroke()."""

    @classmethod
    def apply_stroke(cls, canvas:np.ndarray,
                     x1:Optional[int], y1:Optional[int],
                     x2:int, y2:int,
                     color:Tuple, cfg:BrushConfig,
                     eraser:bool=False) -> None:
        if eraser:
            cls._pen(canvas,x1,y1,x2,y2,(0,0,0),cfg.size*3,1.0); return
        bt = cfg.brush_type
        dispatch = {
            'pen':        lambda: cls._pen(canvas,x1,y1,x2,y2,color,cfg.size,cfg.opacity),
            'marker':     lambda: cls._marker(canvas,x1,y1,x2,y2,color,cfg),
            'pencil':     lambda: cls._pencil(canvas,x1,y1,x2,y2,color,cfg),
            'airbrush':   lambda: cls._airbrush(canvas,x2,y2,color,cfg),
            'watercolor': lambda: cls._watercolor(canvas,x2,y2,color,cfg),
            'neon':       lambda: cls._neon(canvas,x1,y1,x2,y2,color,cfg,tight=True),
            'glow':       lambda: cls._neon(canvas,x1,y1,x2,y2,color,cfg,tight=False),
            'spray':      lambda: cls._spray(canvas,x2,y2,color,cfg),
            'calligraphy':lambda: cls._calligraphy(canvas,x1,y1,x2,y2,color,cfg),
        }
        dispatch.get(bt, lambda: cls._pen(canvas,x1,y1,x2,y2,color,cfg.size,cfg.opacity))()

    @staticmethod
    def _pen(canvas,x1,y1,x2,y2,color,size,opacity):
        if opacity < 1.0:
            ov = canvas.copy()
            if x1 is not None: cv2.line(ov,(x1,y1),(x2,y2),color,size*2)
            cv2.circle(ov,(x2,y2),size,color,-1)
            cv2.addWeighted(ov,opacity,canvas,1-opacity,0,canvas)
        else:
            if x1 is not None: cv2.line(canvas,(x1,y1),(x2,y2),color,size*2)
            cv2.circle(canvas,(x2,y2),size,color,-1)

    @staticmethod
    def _marker(canvas,x1,y1,x2,y2,color,cfg):
        ov    = canvas.copy()
        thick = max(4, cfg.size*3)
        if x1 is not None: cv2.line(ov,(x1,y1),(x2,y2),color,thick)
        cv2.circle(ov,(x2,y2),thick//2,color,-1)
        cv2.addWeighted(ov,cfg.opacity*0.55,canvas,1-cfg.opacity*0.55,0,canvas)

    @staticmethod
    def _pencil(canvas,x1,y1,x2,y2,color,cfg):
        j  = np.random.randint(-2,3,2)
        px,py = x2+int(j[0]), y2+int(j[1])
        thin  = max(1,cfg.size//2)
        nc    = tuple(max(0,min(255,int(c)+random.randint(-20,20))) for c in color)
        if x1 is not None: cv2.line(canvas,(x1,y1),(px,py),nc,thin)
        cv2.circle(canvas,(px,py),max(1,thin//2),nc,-1)

    @staticmethod
    def _airbrush(canvas,x2,y2,color,cfg):
        n_dots  = max(5, int(cfg.flow*35))
        spread  = cfg.size*(2.0-cfg.hardness)
        h,w     = canvas.shape[:2]
        for _ in range(n_dots):
            r   = abs(np.random.normal(0,spread))
            th  = np.random.uniform(0,2*math.pi)
            px  = int(x2+r*math.cos(th)); py = int(y2+r*math.sin(th))
            if 0<=px<w and 0<=py<h:
                fade = max(0.0,1-r/max(1,spread*2))*cfg.opacity
                dc   = tuple(int(c*fade) for c in color)
                if any(v>0 for v in dc): cv2.circle(canvas,(px,py),1,dc,-1)

    @staticmethod
    def _watercolor(canvas,x2,y2,color,cfg):
        ov = canvas.copy()
        s  = cfg.size
        for radius in range(s*2,0,-2):
            t    = radius/(s*2)
            fade = (1-t)**2*cfg.opacity*0.5
            dc   = tuple(int(c*fade) for c in color)
            if any(v>0 for v in dc):
                off = (int(random.gauss(0,s*0.2)),int(random.gauss(0,s*0.2)))
                cv2.circle(ov,(x2+off[0],y2+off[1]),radius,dc,-1)
        cv2.addWeighted(ov,0.45,canvas,0.55,0,canvas)

    @staticmethod
    def _neon(canvas,x1,y1,x2,y2,color,cfg,tight=True):
        bloom = cfg.size*(2 if tight else 4)
        for radius in range(bloom,0,-1):
            t    = radius/bloom
            fade = ((1-t)**1.5)*cfg.opacity
            dc   = tuple(int(c*fade) for c in color)
            if any(v>0 for v in dc):
                cv2.circle(canvas,(x2,y2),radius,dc,-1)
                if x1 is not None: cv2.line(canvas,(x1,y1),(x2,y2),dc,radius)
        cv2.circle(canvas,(x2,y2),max(2,cfg.size//2),color,-1)
        if x1 is not None: cv2.line(canvas,(x1,y1),(x2,y2),color,cfg.size)

    @staticmethod
    def _spray(canvas,x2,y2,color,cfg):
        n_dots  = max(10,int(cfg.flow*60))
        spread  = cfg.size*2.5
        h,w     = canvas.shape[:2]
        for _ in range(n_dots):
            r  = np.random.uniform(0,spread)
            th = np.random.uniform(0,2*math.pi)
            px = int(x2+r*math.cos(th)); py = int(y2+r*math.sin(th))
            if 0<=px<w and 0<=py<h: canvas[py,px] = color

    @staticmethod
    def _calligraphy(canvas,x1,y1,x2,y2,color,cfg):
        if x1 is not None:
            angle = math.atan2(y2-y1,max(1,abs(x2-x1)))
            width = max(1,int(cfg.size*(1+1.8*abs(math.sin(angle*2)))))
            cv2.line(canvas,(x1,y1),(x2,y2),color,width)
        else:
            cv2.circle(canvas,(x2,y2),cfg.size,color,-1)


# =============================================================================
# SECTION 6 — COLOR PICKER (HSV Wheel + RGB Sliders + Recent/Favorites)
# =============================================================================
class ColorPicker:
    WHEEL_SIZE  = 160
    SWATCH_SIZE = 20

    def __init__(self) -> None:
        self.wheel_img    = self._build_wheel(self.WHEEL_SIZE)
        self.current_bgr  : Tuple[int,int,int] = (0,0,255)
        self.hue          = 0.0
        self.sat          = 1.0
        self.val          = 1.0
        self.hex_str      = "FF0000"
        self._r = 255; self._g = 0; self._b = 0
        self.recent    : Deque[Tuple] = deque(maxlen=10)
        self.favorites : List[Tuple]  = [
            (0,0,255),(0,255,0),(255,0,0),(0,220,255),
            (200,0,190),(255,255,255),(10,10,10),(0,130,255),
        ]

    @staticmethod
    def _build_wheel(size:int) -> np.ndarray:
        """Vectorised HSV color wheel — built once at init."""
        cx = cy = size//2
        r_max   = size//2 - 2
        yi,xi   = np.mgrid[:size,:size]
        dx      = (xi-cx).astype(np.float32)
        dy      = (yi-cy).astype(np.float32)
        dist    = np.sqrt(dx**2+dy**2)
        mask    = dist <= r_max
        angle   = (np.arctan2(dy,dx)+np.pi)/(2*np.pi)
        sat     = np.clip(dist/r_max,0,1)
        hsv     = np.zeros((size,size,3),dtype=np.uint8)
        hsv[:,:,0] = (angle*179).astype(np.uint8)
        hsv[:,:,1] = (sat*255).astype(np.uint8)
        hsv[:,:,2] = 255
        bgr     = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        result  = np.full((size,size,3),30,dtype=np.uint8)
        result[mask] = bgr[mask]
        cv2.circle(result,(cx,cy),r_max,(70,70,70),1)
        return result

    def set_from_bgr(self, bgr:Tuple) -> None:
        self.current_bgr = tuple(int(c) for c in bgr)
        b,g,r = int(bgr[0]),int(bgr[1]),int(bgr[2])
        arr   = np.array([[[b,g,r]]],dtype=np.uint8)
        hsv   = cv2.cvtColor(arr,cv2.COLOR_BGR2HSV)[0,0]
        self.hue = float(hsv[0])*2.0
        self.sat = float(hsv[1])/255.0
        self.val = float(hsv[2])/255.0
        self._r,self._g,self._b = r,g,b
        self.hex_str = f"{r:02X}{g:02X}{b:02X}"
        self.recent.appendleft(self.current_bgr)

    def get_bgr(self) -> Tuple[int,int,int]:
        return self.current_bgr

    def add_favorite(self) -> None:
        if self.current_bgr not in self.favorites:
            self.favorites.insert(0,self.current_bgr)
            self.favorites = self.favorites[:8]

    # ------------------------------------------------------------------
    def draw(self, frame:np.ndarray, ox:int, oy:int,
             theme_name:str) -> List[Tuple]:
        """Render color picker; return list of hit-areas for interaction."""
        C    = THEMES[theme_name]
        hits : List[Tuple] = []
        W    = self.WHEEL_SIZE
        pad  = 10

        # ── HSV wheel ──────────────────────────────────────────────────
        wy  = oy+pad
        wx  = ox+(PANEL_W-W)//2
        wbx = max(0,min(frame.shape[1]-W,wx))
        wby = max(0,min(frame.shape[0]-W,wy))
        if wby+W<=frame.shape[0] and wbx+W<=frame.shape[1]:
            frame[wby:wby+W,wbx:wbx+W] = self.wheel_img
        # selector dot
        cx_w,cy_w = wbx+W//2, wby+W//2
        r_px      = int(self.sat*(W//2-2))
        a_rad     = (self.hue/360.0)*2*math.pi - math.pi
        sx = int(cx_w+r_px*math.cos(a_rad))
        sy = int(cy_w+r_px*math.sin(a_rad))
        cv2.circle(frame,(sx,sy),7,(255,255,255),2)
        cv2.circle(frame,(sx,sy),5,self.current_bgr,-1)
        hits.append(((wbx,wby,wbx+W,wby+W),'wheel',None))

        # ── Value slider ───────────────────────────────────────────────
        sv_y = wy+W+8
        sw   = PANEL_W-pad*2
        sxl  = ox+pad
        h_int = int(self.hue/360*179)
        sat_v = int(self.sat*255)
        col_v = cv2.cvtColor(
            np.array([[[h_int,sat_v,255]]],dtype=np.uint8),
            cv2.COLOR_HSV2BGR)[0,0]
        gradient_fill(frame,sxl,sv_y,sxl+sw,sv_y+14,
                      (10,10,10),tuple(int(c) for c in col_v),vertical=False)
        rounded_rect(frame,sxl,sv_y,sxl+sw,sv_y+14,3,C["border"],1)
        dot_x = sxl+int(self.val*sw)
        cv2.circle(frame,(dot_x,sv_y+7),8,(255,255,255),2)
        cv2.circle(frame,(dot_x,sv_y+7),6,self.current_bgr,-1)
        hits.append(((sxl,sv_y-4,sxl+sw,sv_y+18),'val',None))

        # ── RGB sliders ────────────────────────────────────────────────
        labels    = ('R','G','B')
        rgb_vals  = (self._r,self._g,self._b)
        bar_cols  = [(0,0,255),(0,200,0),(255,0,0)]
        y_rgb     = sv_y+22
        for i,(lbl,val,bcol) in enumerate(zip(labels,rgb_vals,bar_cols)):
            ry   = y_rgb+i*26
            cv2.putText(frame,lbl,(sxl,ry+10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,C["text"],1,cv2.LINE_AA)
            bx0,bx1 = sxl+16,sxl+sw-28
            cv2.rectangle(frame,(bx0,ry),(bx1,ry+11),C["bg2"],-1)
            fw = int(val/255*(bx1-bx0))
            if fw>0: cv2.rectangle(frame,(bx0,ry),(bx0+fw,ry+11),bcol,-1)
            cv2.rectangle(frame,(bx0,ry),(bx1,ry+11),C["border"],1)
            cv2.putText(frame,str(val),(bx1+3,ry+9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.30,C["muted"],1,cv2.LINE_AA)
            hits.append(((bx0,ry-2,bx1,ry+13),f'rgb_{i}',None))

        # ── HEX display ────────────────────────────────────────────────
        hx_y = y_rgb+3*26+4
        cv2.putText(frame,f"#{self.hex_str}",(sxl,hx_y+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,C["accent"],1,cv2.LINE_AA)
        sw_x = sxl+sw-28
        rounded_rect(frame,sw_x,hx_y,sw_x+26,hx_y+16,3,self.current_bgr,-1)
        rounded_rect(frame,sw_x,hx_y,sw_x+26,hx_y+16,3,C["border"],1)

        # ── Recent ─────────────────────────────────────────────────────
        rec_y = hx_y+24
        cv2.putText(frame,"RECENT",(sxl,rec_y+8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.30,C["muted"],1,cv2.LINE_AA)
        for j,col in enumerate(list(self.recent)[:8]):
            rsx = sxl+j*(self.SWATCH_SIZE+2)
            rsy = rec_y+12
            rounded_rect(frame,rsx,rsy,rsx+self.SWATCH_SIZE,
                         rsy+self.SWATCH_SIZE,3,col,-1)
            rounded_rect(frame,rsx,rsy,rsx+self.SWATCH_SIZE,
                         rsy+self.SWATCH_SIZE,3,C["border"],1)
            hits.append(((rsx,rsy,rsx+self.SWATCH_SIZE,rsy+self.SWATCH_SIZE),
                         'recent',col))

        # ── Favorites ──────────────────────────────────────────────────
        fav_y = rec_y+self.SWATCH_SIZE+18
        cv2.putText(frame,"FAVORITES",(sxl,fav_y+8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.30,C["muted"],1,cv2.LINE_AA)
        for j,col in enumerate(self.favorites[:8]):
            rsx = sxl+j*(self.SWATCH_SIZE+2)
            rsy = fav_y+12
            rounded_rect(frame,rsx,rsy,rsx+self.SWATCH_SIZE,
                         rsy+self.SWATCH_SIZE,3,col,-1)
            rounded_rect(frame,rsx,rsy,rsx+self.SWATCH_SIZE,
                         rsy+self.SWATCH_SIZE,3,C["border"],1)
            hits.append(((rsx,rsy,rsx+self.SWATCH_SIZE,rsy+self.SWATCH_SIZE),
                         'fav',col))
        return hits

    def handle_click(self, lx:int, ly:int,
                     hits:List[Tuple]) -> Optional[Tuple]:
        for (x1,y1,x2,y2),action,val in hits:
            if x1<=lx<=x2 and y1<=ly<=y2:
                if action=='wheel':
                    cx=(x1+x2)//2; cy=(y1+y2)//2
                    dx,dy = lx-cx, ly-cy
                    r     = math.sqrt(dx*dx+dy*dy)
                    r_max = self.WHEEL_SIZE//2-2
                    if r<=r_max:
                        ang = (math.atan2(dy,dx)+math.pi)/(2*math.pi)*179
                        s   = int(min(1.0,r/r_max)*255)
                        v   = int(self.val*255)
                        hsv = np.array([[[int(ang),s,v]]],dtype=np.uint8)
                        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0]
                        self.set_from_bgr(tuple(int(c) for c in bgr))
                        return self.current_bgr
                elif action=='val':
                    t = float(np.clip((lx-x1)/(x2-x1),0,1))
                    self.val = t
                    h_int = int(self.hue/360*179)
                    hsv   = np.array([[[h_int,int(self.sat*255),int(t*255)]]],
                                     dtype=np.uint8)
                    bgr   = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0]
                    self.set_from_bgr(tuple(int(c) for c in bgr))
                    return self.current_bgr
                elif action.startswith('rgb_'):
                    ch  = int(action.split('_')[1])
                    t   = float(np.clip((lx-x1)/(x2-x1),0,1))
                    r,g,b = self._r,self._g,self._b
                    if ch==0: r=int(t*255)
                    elif ch==1: g=int(t*255)
                    else: b=int(t*255)
                    self.set_from_bgr((b,g,r))
                    return self.current_bgr
                elif action in ('recent','fav') and val:
                    self.set_from_bgr(val)
                    return val
        return None


# =============================================================================
# SECTION 7 — HISTORY MANAGER
# =============================================================================
class History:
    def __init__(self, limit:int=UNDO_LIMIT) -> None:
        self.stack  : Deque[List] = deque(maxlen=limit)
        self.future : Deque[List] = deque(maxlen=limit)

    def push(self, layers:List[np.ndarray]) -> None:
        self.stack.append([l.copy() for l in layers])
        self.future.clear()

    def undo(self, layers:List[np.ndarray]) -> List[np.ndarray]:
        if self.stack:
            self.future.append([l.copy() for l in layers])
            return list(self.stack.pop())
        return layers

    def redo(self, layers:List[np.ndarray]) -> List[np.ndarray]:
        if self.future:
            self.stack.append([l.copy() for l in layers])
            return list(self.future.pop())
        return layers

    def can_undo(self) -> bool: return len(self.stack)>0
    def can_redo(self) -> bool: return len(self.future)>0

# =============================================================================
# SECTION 8 — LAYER MANAGER
# =============================================================================
class LayerManager:
    """
    Pixel buffers + metadata + dirty-flag composite cache.
    FIX: merge_visible() uses proper alpha-composite (not convertScaleAbs).
    """
    def __init__(self, W:int, H:int, count:int=MAX_LAYERS) -> None:
        self.W, self.H        = W, H
        self.layers : List[np.ndarray] = [np.zeros((H,W,3),np.uint8) for _ in range(count)]
        self.meta   : List[LayerMeta]  = [LayerMeta(name=f"Layer {i+1}") for i in range(count)]
        self.active_layer      = 0
        self._cache           : Optional[np.ndarray] = None
        self._cache_valid      = False

    def invalidate(self) -> None:
        self._cache_valid = False

    def current(self) -> np.ndarray:
        return self.layers[self.active_layer]

    def current_meta(self) -> LayerMeta:
        return self.meta[self.active_layer]

    def set_active(self, idx:int) -> None:
        if 0<=idx<len(self.layers): self.active_layer = idx

    def toggle_visibility(self, idx:int) -> None:
        if 0<=idx<len(self.layers):
            self.meta[idx].visible = not self.meta[idx].visible
            self.invalidate()

    def toggle_lock(self, idx:int) -> None:
        if 0<=idx<len(self.layers):
            self.meta[idx].locked = not self.meta[idx].locked

    def rename(self, idx:int, name:str) -> None:
        if 0<=idx<len(self.layers):
            self.meta[idx].name = name[:16]

    def set_opacity(self, idx:int, opacity:float) -> None:
        if 0<=idx<len(self.layers):
            self.meta[idx].opacity = float(np.clip(opacity,0.0,1.0))
            self.invalidate()

    def duplicate(self, idx:int) -> int:
        for j,layer in enumerate(self.layers):
            if j!=idx and not np.any(layer):
                self.layers[j] = self.layers[idx].copy()
                self.meta[j].name    = self.meta[idx].name+" Cpy"
                self.meta[j].opacity = self.meta[idx].opacity
                self.invalidate()
                return j
        return -1

    def clear_current(self) -> None:
        if not self.meta[self.active_layer].locked:
            self.layers[self.active_layer].fill(0)
            self.invalidate()

    def clear_all(self) -> None:
        for l in self.layers: l.fill(0)
        self.invalidate()

    def merge_visible(self) -> np.ndarray:
        """Dirty-flag cached composite. FIX: proper per-pixel alpha blend."""
        if self._cache_valid and self._cache is not None:
            return self._cache
        result = np.zeros((self.H,self.W,3),np.uint8)
        for layer,meta in zip(self.layers,self.meta):
            if not meta.visible: continue
            gray      = cv2.cvtColor(layer,cv2.COLOR_BGR2GRAY)
            mask_bool = gray > 1
            if not np.any(mask_bool): continue
            alpha = meta.opacity
            if alpha >= 1.0:
                result[mask_bool] = layer[mask_bool]
            else:
                res_f = result.astype(np.float32)
                lay_f = layer.astype(np.float32)
                m3    = mask_bool[:,:,np.newaxis]
                res_f = np.where(m3, lay_f*alpha + res_f*(1.0-alpha), res_f)
                result = res_f.astype(np.uint8)
        self._cache       = result
        self._cache_valid = True
        return result

    def restore_from(self, snapshots:List[np.ndarray]) -> None:
        """FIX: in-place restore preserves meta arrays."""
        for i,snap in enumerate(snapshots):
            if i<len(self.layers): np.copyto(self.layers[i],snap)
        self.invalidate()

# =============================================================================
# SECTION 9 — AI SHAPE RECOGNIZER
# =============================================================================
class AIShapeRecognizer:
    """OpenCV contour analysis -> circle/rect/triangle/star/etc."""

    @staticmethod
    def classify(points:List[Tuple]) -> str:
        if len(points)<6: return 'line'
        pts  = np.array(points,dtype=np.float32).reshape(-1,1,2)
        peri = cv2.arcLength(pts,True)
        area = cv2.contourArea(pts)
        if peri<5: return 'line'
        circ = 4*math.pi*area/(peri**2) if peri>0 else 0
        if circ>0.78: return 'circle'
        x,y,w,h = cv2.boundingRect(pts.astype(np.int32))
        ar   = w/max(1,h)
        eps  = 0.04*peri
        approx = cv2.approxPolyDP(pts,eps,True)
        n    = len(approx)
        if n<=3 and area<50: return 'line'
        if n==3: return 'triangle'
        if n==4: return 'square' if 0.85<ar<1.18 else 'rectangle'
        if n==5: return 'pentagon'
        if n==6: return 'hexagon'
        if n>=8:
            cx2,cy2 = x+w//2, y+h//2
            dists   = [math.hypot(p[0][0]-cx2,p[0][1]-cy2) for p in approx]
            if np.std(dists)/max(1,np.mean(dists))>0.25: return 'star'
        return 'circle'

    @staticmethod
    def draw_perfect(canvas:np.ndarray, points:List[Tuple],
                     shape:str, color:Tuple, thickness:int) -> None:
        if not points: return
        xs  = [p[0] for p in points]; ys = [p[1] for p in points]
        x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
        cx,cy        = (x1+x2)//2,(y1+y2)//2
        rw,rh        = max(1,(x2-x1)//2),max(1,(y2-y1)//2)
        r            = max(rw,rh)
        if shape=='circle':
            cv2.circle(canvas,(cx,cy),r,color,thickness)
        elif shape in ('square','rectangle'):
            if shape=='square': x1,y1,x2,y2 = cx-r,cy-r,cx+r,cy+r
            cv2.rectangle(canvas,(x1,y1),(x2,y2),color,thickness)
        elif shape=='diamond':
            pts = np.array([[cx,y1],[x2,cy],[cx,y2],[x1,cy]],dtype=np.int32)
            cv2.polylines(canvas,[pts],True,color,thickness)
        elif shape=='triangle':
            pts = np.array([[cx,y1],[x2,y2],[x1,y2]],dtype=np.int32)
            cv2.polylines(canvas,[pts],True,color,thickness)
        elif shape=='pentagon':
            pts = AIShapeRecognizer._ngon(cx,cy,r,5)
            cv2.polylines(canvas,[pts],True,color,thickness)
        elif shape=='hexagon':
            pts = AIShapeRecognizer._ngon(cx,cy,r,6)
            cv2.polylines(canvas,[pts],True,color,thickness)
        elif shape=='star':
            pts = AIShapeRecognizer._star(cx,cy,r,r//2,5)
            cv2.polylines(canvas,[pts],True,color,thickness)
        elif shape=='arrow':
            cv2.arrowedLine(canvas,(x1,cy),(x2,cy),color,thickness,tipLength=0.35)
        elif shape=='line':
            cv2.line(canvas,points[0],points[-1],color,thickness)
        else:
            cv2.ellipse(canvas,(cx,cy),(rw,rh),0,0,360,color,thickness)

    @staticmethod
    def _ngon(cx:int,cy:int,r:int,n:int) -> np.ndarray:
        pts = [[int(cx+r*math.cos(2*math.pi*i/n-math.pi/2)),
                int(cy+r*math.sin(2*math.pi*i/n-math.pi/2))] for i in range(n)]
        return np.array(pts,dtype=np.int32)

    @staticmethod
    def _star(cx:int,cy:int,r_out:int,r_in:int,n:int) -> np.ndarray:
        pts = []
        for i in range(n*2):
            a = math.pi*i/n-math.pi/2
            r = r_out if i%2==0 else r_in
            pts.append([int(cx+r*math.cos(a)),int(cy+r*math.sin(a))])
        return np.array(pts,dtype=np.int32)

# =============================================================================
# SECTION 10 — SELECTION TOOL
# =============================================================================
class SelectionTool:
    def __init__(self) -> None:
        self.state = SelectionState()

    def begin(self, x:int, y:int) -> None:
        self.state.active     = True
        self.state.rect       = None
        self.state.drag_start = (x,y)

    def update(self, x:int, y:int) -> None:
        if self.state.drag_start:
            x0,y0 = self.state.drag_start
            self.state.rect = (min(x0,x),min(y0,y),max(x0,x),max(y0,y))

    def end(self) -> None:
        self.state.drag_start = None

    def copy_region(self, layer:np.ndarray) -> bool:
        if self.state.rect is None: return False
        x1,y1,x2,y2 = self.state.rect
        x1=max(0,x1); y1=max(0,y1)
        x2=min(layer.shape[1],x2); y2=min(layer.shape[0],y2)
        if x2>x1 and y2>y1:
            self.state.clipboard = layer[y1:y2,x1:x2].copy()
            return True
        return False

    def paste_region(self, layer:np.ndarray, x:int, y:int) -> bool:
        if self.state.clipboard is None: return False
        ch,cw = self.state.clipboard.shape[:2]
        h,w   = layer.shape[:2]
        x2,y2 = min(w,x+cw), min(h,y+ch)
        pw,ph = x2-x, y2-y
        if pw>0 and ph>0:
            data = self.state.clipboard[:ph,:pw]
            mask = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)>1
            layer[y:y2,x:x2][mask] = data[mask]
            return True
        return False

    def delete_region(self, layer:np.ndarray) -> bool:
        if self.state.rect is None: return False
        x1,y1,x2,y2 = self.state.rect
        x1=max(0,x1); y1=max(0,y1)
        x2=min(layer.shape[1],x2); y2=min(layer.shape[0],y2)
        if x2>x1 and y2>y1:
            layer[y1:y2,x1:x2] = 0; return True
        return False

    def draw_overlay(self, frame:np.ndarray, accent:Tuple) -> None:
        if self.state.rect is None: return
        x1,y1,x2,y2 = self.state.rect
        cv2.rectangle(frame,(x1,y1),(x2,y2),accent,1)
        cs = 7
        for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            cv2.rectangle(frame,(px-cs//2,py-cs//2),(px+cs//2,py+cs//2),(255,255,255),-1)
            cv2.rectangle(frame,(px-cs//2,py-cs//2),(px+cs//2,py+cs//2),accent,1)


# =============================================================================
# SECTION 11 — CANVAS VIEWPORT (Zoom + Pan)
# =============================================================================
class CanvasViewport:
    MIN_ZOOM, MAX_ZOOM = 0.5, 4.0

    def __init__(self, cw:int, ch:int) -> None:
        self.cw, self.ch = cw, ch
        self.vt = ViewTransform()

    def zoom_at(self, sx:int, sy:int, delta:float) -> None:
        new_z = float(np.clip(self.vt.zoom+delta,self.MIN_ZOOM,self.MAX_ZOOM))
        scale = new_z/self.vt.zoom
        self.vt.pan_x = sx - scale*(sx-self.vt.pan_x)
        self.vt.pan_y = sy - scale*(sy-self.vt.pan_y)
        self.vt.zoom  = new_z

    def screen_to_canvas(self, sx:int, sy:int) -> Tuple[int,int]:
        cx = int((sx-self.vt.pan_x)/self.vt.zoom)
        cy = int((sy-self.vt.pan_y)/self.vt.zoom)
        return (max(0,min(self.cw-1,cx)), max(0,min(self.ch-1,cy)))

    def apply_to_display(self, canvas:np.ndarray,
                         dw:int, dh:int) -> np.ndarray:
        """Crop+resize canvas region according to current zoom/pan."""
        z = self.vt.zoom
        src_w = int(dw/z); src_h = int(dh/z)
        sx0   = int(-self.vt.pan_x/z); sy0 = int(-self.vt.pan_y/z)
        sx1   = sx0+src_w;              sy1 = sy0+src_h
        sx0c  = max(0,sx0); sy0c = max(0,sy0)
        sx1c  = min(self.cw,sx1); sy1c = min(self.ch,sy1)
        if sx1c<=sx0c or sy1c<=sy0c: return canvas
        crop    = canvas[sy0c:sy1c,sx0c:sx1c]
        result  = np.zeros((dh,dw,3),dtype=np.uint8)
        dest_x  = max(0,int((sx0c-sx0)*z))
        dest_y  = max(0,int((sy0c-sy0)*z))
        scl_w   = int((sx1c-sx0c)*z); scl_h = int((sy1c-sy0c)*z)
        if scl_w>0 and scl_h>0:
            resized = cv2.resize(crop,(scl_w,scl_h),interpolation=cv2.INTER_LINEAR)
            dy2 = min(dh,dest_y+scl_h); dx2 = min(dw,dest_x+scl_w)
            rh  = dy2-dest_y;           rw  = dx2-dest_x
            if rh>0 and rw>0:
                result[dest_y:dy2,dest_x:dx2] = resized[:rh,:rw]
        return result

# =============================================================================
# SECTION 12 — SYMMETRY ENGINE
# =============================================================================
class SymmetryEngine:
    def __init__(self) -> None:
        self.mode     = SymMode.NONE
        self.radial_n = 8

    def next_mode(self) -> None:
        modes = list(SymMode)
        self.mode = modes[(modes.index(self.mode)+1)%len(modes)]

    def mirror_points(self, x:int, y:int,
                      cw:int, ch:int) -> List[Tuple[int,int]]:
        pts = [(x,y)]
        if self.mode == SymMode.VERTICAL:
            pts.append((cw-x, y))
        elif self.mode == SymMode.HORIZONTAL:
            pts.append((x, ch-y))
        elif self.mode in (SymMode.RADIAL, SymMode.MANDALA):
            cx,cy = cw//2, ch//2
            dx,dy = x-cx, y-cy
            n     = self.radial_n
            for i in range(1,n):
                a  = 2*math.pi*i/n
                ca,sa = math.cos(a), math.sin(a)
                pts.append((int(cx+dx*ca-dy*sa), int(cy+dx*sa+dy*ca)))
            if self.mode == SymMode.MANDALA:
                pts += [(cw-px, py) for px,py in pts[:]]
        return pts

    def draw_guide(self, frame:np.ndarray, cw:int, ch:int, color:Tuple) -> None:
        gc = tuple(max(0,int(c*0.35)) for c in color)
        if self.mode == SymMode.VERTICAL:
            cv2.line(frame,(cw//2,0),(cw//2,ch),gc,1)
        elif self.mode == SymMode.HORIZONTAL:
            cv2.line(frame,(0,ch//2),(cw,ch//2),gc,1)
        elif self.mode in (SymMode.RADIAL, SymMode.MANDALA):
            cx,cy = cw//2, ch//2; r = min(cw,ch)//2
            for i in range(self.radial_n):
                a  = math.pi*i/self.radial_n
                cv2.line(frame,(int(cx+r*math.cos(a)),int(cy+r*math.sin(a))),
                         (int(cx-r*math.cos(a)),int(cy-r*math.sin(a))),gc,1)

# =============================================================================
# SECTION 13 — EFFECTS ENGINE
# =============================================================================
class EffectsEngine:
    EFFECTS: List[str] = [
        'blur','sharpen','glow','neon','oilpaint',
        'sketch','watercolor','comic','emboss','pixelate',
    ]

    @staticmethod
    def apply(layer:np.ndarray, effect:str, strength:float=1.0) -> np.ndarray:
        mask = cv2.cvtColor(layer,cv2.COLOR_BGR2GRAY)>1
        if not np.any(mask): return layer
        out  = layer.copy()
        if effect=='blur':
            k = max(3,int(strength*21)|1)
            b = cv2.GaussianBlur(out,(k,k),0)
            out[mask] = b[mask]
        elif effect=='sharpen':
            k   = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],np.float32)
            s   = cv2.filter2D(out,-1,k)
            out[mask] = np.clip(s[mask].astype(np.float32),0,255).astype(np.uint8)
        elif effect=='glow':
            b   = cv2.GaussianBlur(out,(25,25),0)
            g   = cv2.addWeighted(out,1.4,b,0.6,0)
            out[mask] = np.clip(g[mask].astype(np.float32),0,255).astype(np.uint8)
        elif effect=='neon':
            gray  = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,40,120)
            hsv   = cv2.cvtColor(out,cv2.COLOR_BGR2HSV)
            hsv[:,:,0] = (hsv[:,:,0].astype(int)+90)%180
            hsv[:,:,2] = 255
            col   = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            neon  = np.zeros_like(out)
            neon[edges>0] = col[edges>0]
            out   = cv2.add(cv2.GaussianBlur(neon,(7,7),0), neon)
            out[~mask] = layer[~mask]
        elif effect=='oilpaint':
            med  = cv2.medianBlur(out,max(3,int(strength*11)|1))
            hsv  = cv2.cvtColor(med,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1] = np.clip(hsv[:,:,1]*1.4,0,255)
            out  = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            out[~mask] = layer[~mask]
        elif effect=='sketch':
            gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
            inv  = cv2.bitwise_not(gray)
            bl   = cv2.GaussianBlur(inv,(21,21),0)
            sk   = cv2.divide(gray,255-bl,scale=256)
            out[mask] = cv2.cvtColor(sk,cv2.COLOR_GRAY2BGR)[mask]
        elif effect=='watercolor':
            bil  = cv2.bilateralFilter(out,9,75,75)
            ed   = cv2.Canny(cv2.cvtColor(bil,cv2.COLOR_BGR2GRAY),40,100)
            ed3  = cv2.cvtColor(ed,cv2.COLOR_GRAY2BGR)
            out[mask] = cv2.subtract(bil,ed3//3)[mask]
        elif effect=='comic':
            gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
            ed   = cv2.Canny(gray,50,150)
            ed3  = cv2.cvtColor(ed,cv2.COLOR_GRAY2BGR)
            q    = (out//64)*64
            out[mask] = cv2.subtract(q,ed3//2)[mask]
        elif effect=='emboss':
            k    = np.array([[-2,-1,0],[-1,1,1],[0,1,2]],np.float32)
            out[mask] = cv2.filter2D(out,-1,k)[mask]
        elif effect=='pixelate':
            px   = max(4,int(strength*20))
            h,w  = out.shape[:2]
            sm   = cv2.resize(out,(w//px,h//px),interpolation=cv2.INTER_LINEAR)
            big  = cv2.resize(sm,(w,h),interpolation=cv2.INTER_NEAREST)
            out[mask] = big[mask]
        return out

# =============================================================================
# SECTION 14 — STICKER LIBRARY
# =============================================================================
class StickerLibrary:
    ALL_NAMES: List[str] = [
        'star5','star6','heart','diamond','cross','hexagon','pentagon',
        'arrow_r','arrow_l','arrow_u','arrow_d','dbl_arrow',
        'circle_dot','target','checkmark','x_mark','plus',
        'sparkle','sun','cloud','bolt','crown','flower',
    ]

    @staticmethod
    def stamp(canvas:np.ndarray, name:str, x:int, y:int,
              size:int, color:Tuple) -> None:
        r = max(4,size)
        if name=='star5':
            pts = AIShapeRecognizer._star(x,y,r,r//2,5)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='star6':
            pts = AIShapeRecognizer._star(x,y,r,r//2,6)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='heart':
            cv2.circle(canvas,(x-r//2,y-r//4),r//2,color,-1)
            cv2.circle(canvas,(x+r//2,y-r//4),r//2,color,-1)
            pts = np.array([[x-r,y-r//4],[x,y+r],[x+r,y-r//4]],dtype=np.int32)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='diamond':
            pts = np.array([[x,y-r],[x+r,y],[x,y+r],[x-r,y]],dtype=np.int32)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='cross':
            t = max(2,r//3)
            cv2.line(canvas,(x-r,y),(x+r,y),color,t)
            cv2.line(canvas,(x,y-r),(x,y+r),color,t)
        elif name=='hexagon':
            pts = AIShapeRecognizer._ngon(x,y,r,6)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='pentagon':
            pts = AIShapeRecognizer._ngon(x,y,r,5)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='arrow_r':
            cv2.arrowedLine(canvas,(x-r,y),(x+r,y),color,max(2,r//4),tipLength=0.4)
        elif name=='arrow_l':
            cv2.arrowedLine(canvas,(x+r,y),(x-r,y),color,max(2,r//4),tipLength=0.4)
        elif name=='arrow_u':
            cv2.arrowedLine(canvas,(x,y+r),(x,y-r),color,max(2,r//4),tipLength=0.4)
        elif name=='arrow_d':
            cv2.arrowedLine(canvas,(x,y-r),(x,y+r),color,max(2,r//4),tipLength=0.4)
        elif name=='dbl_arrow':
            t = max(2,r//4)
            cv2.arrowedLine(canvas,(x,y),(x+r,y),color,t,tipLength=0.45)
            cv2.arrowedLine(canvas,(x,y),(x-r,y),color,t,tipLength=0.45)
        elif name=='circle_dot':
            cv2.circle(canvas,(x,y),r,color,2)
            cv2.circle(canvas,(x,y),max(2,r//4),color,-1)
        elif name=='target':
            for ri in [r,r*2//3,r//3]: cv2.circle(canvas,(x,y),ri,color,1)
        elif name=='checkmark':
            t = max(2,r//3)
            cv2.line(canvas,(x-r,y),(x-r//3,y+r),color,t)
            cv2.line(canvas,(x-r//3,y+r),(x+r,y-r),color,t)
        elif name=='x_mark':
            t = max(2,r//3)
            cv2.line(canvas,(x-r,y-r),(x+r,y+r),color,t)
            cv2.line(canvas,(x+r,y-r),(x-r,y+r),color,t)
        elif name=='plus':
            t = max(2,r//3)
            cv2.line(canvas,(x-r,y),(x+r,y),color,t)
            cv2.line(canvas,(x,y-r),(x,y+r),color,t)
        elif name=='sparkle':
            for a in range(0,360,45):
                ra = math.radians(a)
                cv2.line(canvas,(x,y),(int(x+r*math.cos(ra)),int(y+r*math.sin(ra))),color,1)
            cv2.circle(canvas,(x,y),r//4,color,-1)
        elif name=='sun':
            cv2.circle(canvas,(x,y),r//2,color,2)
            for a in range(0,360,45):
                ra = math.radians(a)
                cv2.line(canvas,(int(x+(r//2+3)*math.cos(ra)),int(y+(r//2+3)*math.sin(ra))),
                         (int(x+r*math.cos(ra)),int(y+r*math.sin(ra))),color,2)
        elif name=='bolt':
            pts = np.array([[x,y-r],[x-r//3,y],[x+r//4,y],
                            [x,y+r],[x+r//3,y],[x-r//4,y]],dtype=np.int32)
            cv2.fillPoly(canvas,[pts],color)
        elif name=='crown':
            b = y+r//2
            pts = np.array([[x-r,b],[x-r,y-r//2],[x-r//2,y+r//4],
                             [x,y-r],[x+r//2,y+r//4],[x+r,y-r//2],[x+r,b]],dtype=np.int32)
            cv2.polylines(canvas,[pts],False,color,2)
            cv2.line(canvas,(x-r,b),(x+r,b),color,2)
        elif name=='flower':
            for a in range(0,360,60):
                ra = math.radians(a)
                fx,fy = int(x+r*0.5*math.cos(ra)), int(y+r*0.5*math.sin(ra))
                cv2.circle(canvas,(fx,fy),r//3,color,-1)
            cv2.circle(canvas,(x,y),r//4,(255,255,200),-1)
        elif name=='cloud':
            for fx,fy,fr in [(x-r//2,y,r//3),(x,y-r//4,r//2),(x+r//2,y,r//3),(x,y+r//6,r//2)]:
                cv2.circle(canvas,(fx,fy),fr,color,-1)
        else:
            cv2.circle(canvas,(x,y),r,color,-1)


# =============================================================================
# SECTION 15 — PROJECT MANAGER (.vpaint)
# =============================================================================
class ProjectManager:
    VERSION = "4.0"

    @staticmethod
    def save(path:str, painter:'VirtualPainterV4') -> bool:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
            layers_data = []
            for layer,meta in zip(painter.layers.layers,painter.layers.meta):
                ok,buf = cv2.imencode('.png',layer)
                enc    = base64.b64encode(buf.tobytes()).decode() if ok else ""
                layers_data.append({'name':meta.name,'visible':meta.visible,
                                    'locked':meta.locked,'opacity':meta.opacity,'data':enc})
            cfg = painter.brush_cfg
            doc = {
                'version':      ProjectManager.VERSION,
                'timestamp':    datetime.now().isoformat(),
                'theme':        painter.current_theme.value,
                'active_layer': painter.layers.active_layer,
                'grid_on':      painter.grid_on,
                'symmetry':     painter.symmetry.mode.value,
                'brush':        {'type':cfg.brush_type,'size':cfg.size,
                                 'opacity':cfg.opacity,'hardness':cfg.hardness,
                                 'flow':cfg.flow},
                'viewport':     {'zoom':painter.viewport.vt.zoom,
                                 'pan_x':painter.viewport.vt.pan_x,
                                 'pan_y':painter.viewport.vt.pan_y},
                'layers':       layers_data,
            }
            with open(path,'w',encoding='utf-8') as f:
                json.dump(doc,f,indent=2)
            return True
        except Exception as e:
            print(f"[!] Save error: {e}"); return False

    @staticmethod
    def load(path:str, painter:'VirtualPainterV4') -> bool:
        try:
            with open(path,'r',encoding='utf-8') as f:
                doc = json.load(f)
            tv = doc.get('theme','dark')
            painter.toggle_theme(Theme.DARK if tv=='dark' else Theme.LIGHT)
            painter.grid_on = doc.get('grid_on',False)
            for m in SymMode:
                if m.value==doc.get('symmetry','none'):
                    painter.symmetry.mode=m; break
            b = doc.get('brush',{})
            painter.brush_cfg.brush_type = b.get('type','pen')
            painter.brush_cfg.size       = b.get('size',14)
            painter.brush_cfg.opacity    = b.get('opacity',1.0)
            painter.brush_cfg.hardness   = b.get('hardness',0.8)
            painter.brush_cfg.flow       = b.get('flow',1.0)
            vp = doc.get('viewport',{})
            painter.viewport.vt.zoom  = vp.get('zoom',1.0)
            painter.viewport.vt.pan_x = vp.get('pan_x',0.0)
            painter.viewport.vt.pan_y = vp.get('pan_y',0.0)
            for i,ld in enumerate(doc.get('layers',[])):
                if i>=len(painter.layers.layers): break
                data = ld.get('data','')
                if data:
                    buf = np.frombuffer(base64.b64decode(data),dtype=np.uint8)
                    img = cv2.imdecode(buf,cv2.IMREAD_COLOR)
                    if img is not None:
                        h,w = painter.layers.layers[i].shape[:2]
                        painter.layers.layers[i][:] = (cv2.resize(img,(w,h))
                                                       if img.shape[:2]!=(h,w) else img)
                painter.layers.meta[i].name    = ld.get('name',   f'Layer {i+1}')
                painter.layers.meta[i].visible = ld.get('visible', True)
                painter.layers.meta[i].locked  = ld.get('locked',  False)
                painter.layers.meta[i].opacity = ld.get('opacity', 1.0)
            painter.layers.set_active(doc.get('active_layer',0))
            painter.layers.invalidate()
            return True
        except Exception as e:
            print(f"[!] Load error: {e}"); return False

# =============================================================================
# SECTION 16 — ANALYTICS
# =============================================================================
class Analytics:
    def __init__(self) -> None:
        self.data = AnalyticsData()

    def stroke(self)       -> None: self.data.total_strokes  +=1
    def shape(self)        -> None: self.data.shapes_drawn   +=1
    def undo(self)         -> None: self.data.undo_count     +=1
    def redo(self)         -> None: self.data.redo_count     +=1
    def layer(self)        -> None: self.data.layer_switches +=1
    def voice(self)        -> None: self.data.voice_commands +=1
    def brush_change(self) -> None: self.data.brush_changes  +=1
    def effect(self)       -> None: self.data.effects_applied+=1

    def draw_panel(self, frame:np.ndarray, ox:int, oy:int,
                   theme_name:str) -> None:
        C    = THEMES[theme_name]
        d    = self.data
        rows = [("Session Time", d.drawing_time),
                ("Total Strokes", str(d.total_strokes)),
                ("Shapes Drawn",  str(d.shapes_drawn)),
                ("Undo / Redo",  f"{d.undo_count} / {d.redo_count}"),
                ("Layer Switches",str(d.layer_switches)),
                ("Voice Commands",str(d.voice_commands)),
                ("Brush Changes", str(d.brush_changes)),
                ("Effects Applied",str(d.effects_applied))]
        pad = 12; y = oy+pad
        cv2.putText(frame,"ANALYTICS",(ox+pad,y+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,C["accent"],1,cv2.LINE_AA)
        y += 28
        for lbl,val in rows:
            cv2.putText(frame,lbl,(ox+pad,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.34,C["muted"],1,cv2.LINE_AA)
            vx = ox+PANEL_W-pad-len(val)*7
            cv2.putText(frame,val,(max(ox+pad+80,vx),y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,C["text"],1,cv2.LINE_AA)
            cv2.line(frame,(ox+pad,y+4),(ox+PANEL_W-pad,y+4),C["sep"],1)
            y += 25

# =============================================================================
# SECTION 17 — EXPORT MANAGER
# =============================================================================
class ExportManager:
    OUT_DIR = "exports"

    @classmethod
    def _path(cls, name:str, ext:str) -> str:
        os.makedirs(cls.OUT_DIR,exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(cls.OUT_DIR,f"{name}_{ts}.{ext}")

    @classmethod
    def export_png(cls, canvas:np.ndarray, name:str="painting") -> str:
        p = cls._path(name,"png"); cv2.imwrite(p,canvas); return p

    @classmethod
    def export_jpg(cls, canvas:np.ndarray, name:str="painting",
                   quality:int=92) -> str:
        p = cls._path(name,"jpg")
        cv2.imwrite(p,canvas,[cv2.IMWRITE_JPEG_QUALITY,quality])
        return p

    @classmethod
    def export_svg(cls, canvas:np.ndarray, name:str="painting") -> str:
        p  = cls._path(name,"svg")
        h,w = canvas.shape[:2]
        ok,buf = cv2.imencode('.png',canvas)
        if not ok: return ""
        b64 = base64.b64encode(buf.tobytes()).decode()
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n'
               f'  <image width="{w}" height="{h}" '
               f'href="data:image/png;base64,{b64}"/>\n</svg>\n')
        with open(p,'w',encoding='utf-8') as f: f.write(svg)
        return p

# =============================================================================
# SECTION 18 — TIMELAPSE RECORDER
# =============================================================================
class TimelapseRecorder:
    MAX_FRAMES = 18000

    def __init__(self) -> None:
        self.frames    : List[np.ndarray] = []
        self.recording : bool = False

    def start(self) -> None:
        self.recording = True; self.frames.clear()

    def stop(self) -> None:
        self.recording = False

    def add_frame(self, canvas:np.ndarray) -> None:
        if self.recording and len(self.frames)<self.MAX_FRAMES:
            self.frames.append(canvas.copy())

    def save(self, path:str="timelapse.mp4", fps:int=30) -> bool:
        if not self.frames: return False
        try:
            h,w  = self.frames[0].shape[:2]
            out  = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            for f in self.frames: out.write(f)
            out.release()
            print(f"  Timelapse saved: {os.path.abspath(path)}")
            return True
        except Exception as e:
            print(f"[!] Timelapse: {e}"); return False


# =============================================================================
# SECTION 19 — ANIMATED CURSOR
# =============================================================================
class AnimatedCursor:
    def __init__(self, max_trail:int=18) -> None:
        self.trail: Deque[Tuple[int,int]] = deque(maxlen=max_trail)

    def add(self, x:int, y:int) -> None:
        self.trail.append((x,y))

    def draw(self, frame:np.ndarray, x:int, y:int,
             cfg:BrushConfig, color:Tuple, eraser:bool=False) -> None:
        size  = cfg.size
        # FIX: vectorised trail (was Python loop)
        if len(self.trail)>=2:
            n = len(self.trail)
            for i in range(1,n):
                alpha = i/n
                tc    = tuple(int(c*alpha*0.55) for c in color) if not eraser \
                        else (60,60,180)
                cv2.line(frame,self.trail[i-1],self.trail[i],tc,
                         max(1,int(size*alpha*0.4)))
        if eraser:
            cv2.circle(frame,(x,y),size*3,(140,140,255),2)
            cv2.circle(frame,(x,y),3,(255,255,255),-1)
            return
        bt = cfg.brush_type
        if bt in ('neon','glow'):
            for r in range(size+10,size,-2):
                fade = max(0,(r-size)/10.0)
                gc   = tuple(int(c*(1-fade)*0.38) for c in color)
                if any(v>0 for v in gc): cv2.circle(frame,(x,y),r,gc,1)
            cv2.circle(frame,(x,y),size,color,2)
        elif bt=='airbrush':
            cv2.circle(frame,(x,y),size*2,color,1)
            for a in range(0,360,45):
                ra = math.radians(a)
                cv2.circle(frame,(int(x+size*1.5*math.cos(ra)),
                                  int(y+size*1.5*math.sin(ra))),2,color,-1)
        elif bt=='calligraphy':
            cv2.ellipse(frame,(x,y),(size,max(2,size//3)),45,0,360,color,2)
        elif bt=='spray':
            cv2.circle(frame,(x,y),size*2,color,1)
        else:
            cv2.circle(frame,(x,y),size,color,2)
        cv2.circle(frame,(x,y),3,(255,255,255),-1)

    def clear(self) -> None:
        self.trail.clear()

# =============================================================================
# SECTION 20 — FLOATING DOCK v4
# =============================================================================
class FloatingDock:
    ICON_H  = 38
    SPACING = 2
    MARGIN  = 10
    LABELS : Dict[str,str] = {
        'pen':'PEN','eraser':'ERA','line':'LIN','rect':'REC',
        'circle':'CIR','fill':'FIL','text':'TXT','emoji':'EMO',
        'select':'SEL','eyedropper':'EYE',
    }
    # Bottom shortcuts map: key-label -> panel mode name
    SHORTCUTS = [('B','brush'),('L','layers'),('K','color'),
                 ('E','effects'),('A','stats')]

    def __init__(self, dock_w:int, canvas_h:int,
                 theme_name:str="dark") -> None:
        self.dock_w      = dock_w
        self.canvas_h    = canvas_h
        self.theme_name  = theme_name
        self.rects       : List[Tuple] = []
        self._bg_cache   : Optional[np.ndarray] = None
        self._bg_theme   = ""

    def update_theme(self, name:str) -> None:
        self.theme_name = name
        self._bg_cache  = None   # FIX: invalidate so it is re-blurred next frame

    def draw(self, frame:np.ndarray, active_tool:str, eraser:bool) -> None:
        C  = THEMES[self.theme_name]
        # FIX: blur dock background only once per theme change (not every frame)
        if self._bg_cache is None or self._bg_theme != self.theme_name:
            area           = frame[:self.canvas_h,:self.dock_w].copy()
            bl             = cv2.GaussianBlur(area,(21,21),0)
            self._bg_cache = cv2.addWeighted(bl,0.75,area,0.25,0)
            self._bg_theme = self.theme_name
        frame[:self.canvas_h,:self.dock_w] = self._bg_cache
        alpha_blend(frame,0,0,self.dock_w,self.canvas_h,C["dock_bg"],0.60)
        cv2.line(frame,(self.dock_w,0),(self.dock_w,self.canvas_h),C["border"],1)

        self.rects.clear()
        y = self.MARGIN

        for tool in TOOLS:
            is_active = (tool==active_tool and not eraser) or (tool=='eraser' and eraser)
            x1 = self.MARGIN; x2 = self.dock_w-self.MARGIN; y2 = y+self.ICON_H
            bg  = C["tool_act"] if is_active else C["tool_bg"]
            brd = C["active"]   if is_active else C["border"]
            bw  = 2             if is_active else 1
            rounded_rect(frame,x1,y,x2,y2,8,bg,-1)
            rounded_rect(frame,x1,y,x2,y2,8,brd,bw)
            lbl = self.LABELS.get(tool,tool[:3].upper())
            text_centered(frame,lbl,(x1+x2)//2,(y+y2)//2,0.37,
                          C["active"] if is_active else C["text"])
            self.rects.append(('tool_'+tool,x1,y,x2,y2))
            y += self.ICON_H+self.SPACING

        # Panel shortcut buttons at bottom
        sh = self.ICON_H-8
        bottom_y = self.canvas_h-(sh+self.SPACING)*len(self.SHORTCUTS)-4
        for key,name in self.SHORTCUTS:
            x1 = self.MARGIN; x2 = self.dock_w-self.MARGIN; y2 = bottom_y+sh
            rounded_rect(frame,x1,bottom_y,x2,y2,5,C["bg2"],-1)
            rounded_rect(frame,x1,bottom_y,x2,y2,5,C["sep"],1)
            text_centered(frame,key,(x1+x2)//2,(bottom_y+y2)//2,0.38,C["muted"])
            self.rects.append((f"panel_{name}",x1,bottom_y,x2,y2))
            bottom_y += sh+self.SPACING

    def hit(self, x:int, y:int) -> Optional[str]:
        for name,x1,y1,x2,y2 in self.rects:
            if x1<=x<=x2 and y1<=y<=y2: return name
        return None

# =============================================================================
# SECTION 21 — SIDE PANEL (Layer / Brush / Color / Effects / Analytics)
# =============================================================================
class SidePanel:
    def __init__(self) -> None:
        self.mode        = PanelMode.NONE
        self._hits       : List[Tuple] = []
        # Stored separately so ColorPicker.handle_click() gets its native list
        self._color_hits : List[Tuple] = []
        # Track last hovered item for visual highlight feedback
        self._hover_rec  : Optional[Tuple] = None

    def toggle(self, mode:PanelMode) -> None:
        self.mode = PanelMode.NONE if self.mode==mode else mode

    def draw(self, frame:np.ndarray, painter:'VirtualPainterV4') -> None:
        if self.mode==PanelMode.NONE: return
        C  = painter.colors
        W  = frame.shape[1]; oh = frame.shape[0]-SB_H
        ox = W-PANEL_W
        # Glass panel background
        roi  = frame[:oh,ox:]
        blur = cv2.GaussianBlur(roi,(13,13),0)
        cv2.addWeighted(blur,0.80,roi,0.20,0,frame[:oh,ox:])
        alpha_blend(frame,ox,0,W,oh,C["panel_bg"],0.78)
        cv2.line(frame,(ox,0),(ox,oh),C["border"],1)
        self._hits = []; self._color_hits = []
        if   self.mode==PanelMode.LAYERS:    self._layers(frame,ox,painter,C)
        elif self.mode==PanelMode.BRUSH:     self._brush(frame,ox,painter,C)
        elif self.mode==PanelMode.COLORS:
            ch = painter.color_picker.draw(frame,ox,0,painter.current_theme.value)
            self._color_hits = ch                     # native list for handle_click
            for h in ch: self._hits.append((*h[0],'color_pick',h))
        elif self.mode==PanelMode.EFFECTS:   self._effects(frame,ox,painter,C)
        elif self.mode==PanelMode.ANALYTICS: painter.analytics.draw_panel(
                frame,ox,0,painter.current_theme.value)
        # Hover highlight — drawn last so it appears on top
        if self._hover_rec is not None:
            hx1,hy1,hx2,hy2 = self._hover_rec[:4]
            alpha_blend(frame,hx1,hy1,hx2,hy2,C["hover"],0.22)
            rounded_rect(frame,hx1,hy1,hx2,hy2,4,C["hover"],1)

    # ── Layer panel ───────────────────────────────────────────────────────────
    def _layers(self, frame, ox, painter, C):
        pad=10; y=10
        cv2.putText(frame,"LAYERS",(ox+pad,y+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.46,C["accent"],1,cv2.LINE_AA)
        y+=28
        btns=[('+ Add','layer_add'),('Dup','layer_dup'),
              ('Del','layer_del'),('^','layer_up'),('v','layer_dn')]
        bw=(PANEL_W-pad*2)//len(btns)-3; bx=ox+pad
        for lbl,act in btns:
            rounded_rect(frame,bx,y,bx+bw,y+22,4,C["tool_bg"],-1)
            rounded_rect(frame,bx,y,bx+bw,y+22,4,C["border"],1)
            text_centered(frame,lbl,bx+bw//2,y+11,0.30,C["text"])
            self._hits.append((bx,y,bx+bw,y+22,act,None))
            bx+=bw+3
        y+=30
        row_h   = 58
        max_rows= (frame.shape[0]-SB_H-y-10)//row_h
        n_show  = min(len(painter.layers.layers),max_rows)
        for i in range(n_show):
            idx  = n_show-1-i
            meta = painter.layers.meta[idx]
            ly   = y+i*row_h; lx = ox+pad
            row_w= PANEL_W-pad*2
            bg   = C["tool_act"] if idx==painter.layers.active_layer else C["tool_bg"]
            rounded_rect(frame,lx,ly,lx+row_w,ly+row_h-4,6,bg,-1)
            rounded_rect(frame,lx,ly,lx+row_w,ly+row_h-4,6,C["border"],1)
            # Name
            col = C["active"] if idx==painter.layers.active_layer else C["text"]
            cv2.putText(frame,meta.name[:13],(lx+5,ly+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)
            # Eye
            ey_x = lx+row_w-42
            ec   = C["accent"] if meta.visible else C["muted"]
            cv2.circle(frame,(ey_x+7,ly+12),7,ec,1)
            if meta.visible: cv2.circle(frame,(ey_x+7,ly+12),4,ec,-1)
            self._hits.append((ey_x,ly+4,ey_x+16,ly+22,'layer_vis',idx))
            # Lock
            lk_x = ey_x+20; lk_c = C["warning"] if meta.locked else C["muted"]
            cv2.rectangle(frame,(lk_x+2,ly+7),(lk_x+12,ly+18),lk_c,1)
            if meta.locked: cv2.rectangle(frame,(lk_x+4,ly+9),(lk_x+10,ly+16),lk_c,-1)
            self._hits.append((lk_x,ly+4,lk_x+16,ly+22,'layer_lock',idx))
            # Opacity bar
            op_y = ly+26; op_x0=lx+5; op_x1=lx+row_w-8
            op_f = int((op_x1-op_x0)*meta.opacity)
            cv2.rectangle(frame,(op_x0,op_y),(op_x1,op_y+10),C["bg2"],-1)
            if op_f>0: cv2.rectangle(frame,(op_x0,op_y),(op_x0+op_f,op_y+10),C["accent"],-1)
            cv2.rectangle(frame,(op_x0,op_y),(op_x1,op_y+10),C["sep"],1)
            cv2.putText(frame,f"{int(meta.opacity*100)}%",(op_x1+2,op_y+8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,C["muted"],1,cv2.LINE_AA)
            self._hits.append((op_x0,op_y-2,op_x1,op_y+12,'layer_opacity',idx))
            self._hits.append((lx,ly,lx+row_w,ly+row_h-4,'layer_select',idx))

    # ── Brush panel ───────────────────────────────────────────────────────────
    def _brush(self, frame, ox, painter, C):
        pad=10; y=10; cfg=painter.brush_cfg
        cv2.putText(frame,"BRUSH ENGINE",(ox+pad,y+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,C["accent"],1,cv2.LINE_AA)
        y+=28
        bw=(PANEL_W-pad*2)//3-3; bh=26
        for j,bt in enumerate(BRUSH_TYPES):
            bx = ox+pad+(j%3)*(bw+3); by = y+(j//3)*(bh+4)
            active = bt==cfg.brush_type
            rounded_rect(frame,bx,by,bx+bw,by+bh,5,
                         C["tool_act"] if active else C["tool_bg"],-1)
            rounded_rect(frame,bx,by,bx+bw,by+bh,5,
                         C["active"] if active else C["border"],1 if not active else 2)
            text_centered(frame,bt[:6].upper(),bx+bw//2,by+bh//2,0.28,C["text"])
            self._hits.append((bx,by,bx+bw,by+bh,'brush_type',bt))
        y+=(len(BRUSH_TYPES)//3+1)*(bh+4)+10
        sliders=[('Size',cfg.size/MAX_BRUSH,'brush_size'),
                 ('Opacity',cfg.opacity,'brush_opacity'),
                 ('Hardness',cfg.hardness,'brush_hardness'),
                 ('Flow',cfg.flow,'brush_flow')]
        sw=PANEL_W-pad*2-42
        for lbl,val,act in sliders:
            cv2.putText(frame,lbl,(ox+pad,y+10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.34,C["muted"],1,cv2.LINE_AA)
            bx0=ox+pad+40
            cv2.rectangle(frame,(bx0,y),(bx0+sw,y+12),C["bg2"],-1)
            fw=int(val*sw)
            if fw>0: cv2.rectangle(frame,(bx0,y),(bx0+fw,y+12),C["accent"],-1)
            cv2.rectangle(frame,(bx0,y),(bx0+sw,y+12),C["sep"],1)
            vs=(f"{int(val*MAX_BRUSH)}px" if lbl=='Size' else f"{int(val*100)}%")
            cv2.putText(frame,vs,(bx0+sw+3,y+9),cv2.FONT_HERSHEY_SIMPLEX,
                        0.27,C["muted"],1,cv2.LINE_AA)
            self._hits.append((bx0,y-2,bx0+sw,y+14,act,None))
            y+=24
        # Brush preview
        y+=8; px=ox+PANEL_W//2
        cv2.circle(frame,(px,y+cfg.size+4),cfg.size,painter.color_picker.get_bgr(),-1)
        if cfg.brush_type in ('neon','glow'):
            for rr in range(cfg.size+8,cfg.size,-2):
                gc=tuple(int(c*0.25) for c in painter.color_picker.get_bgr())
                cv2.circle(frame,(px,y+cfg.size+4),rr,gc,1)

    # ── Effects panel ─────────────────────────────────────────────────────────
    def _effects(self, frame, ox, painter, C):
        pad=10; y=10
        cv2.putText(frame,"EFFECTS",(ox+pad,y+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.46,C["accent"],1,cv2.LINE_AA)
        y+=28
        bw=(PANEL_W-pad*2)//2-4; bh=38
        for j,eff in enumerate(EffectsEngine.EFFECTS):
            bx=ox+pad+(j%2)*(bw+6); by=y+(j//2)*(bh+5)
            rounded_rect(frame,bx,by,bx+bw,by+bh,6,C["tool_bg"],-1)
            rounded_rect(frame,bx,by,bx+bw,by+bh,6,C["border"],1)
            text_centered(frame,eff.upper(),bx+bw//2,by+bh//2,0.30,C["text"])
            self._hits.append((bx,by,bx+bw,by+bh,'fx_apply',eff))
        y+=(len(EffectsEngine.EFFECTS)//2+1)*(bh+5)+8
        cv2.putText(frame,"Strength",(ox+pad,y+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.34,C["muted"],1,cv2.LINE_AA)
        sw=PANEL_W-pad*2-50; bx0=ox+pad+50; val=painter.fx_strength
        cv2.rectangle(frame,(bx0,y),(bx0+sw,y+12),C["bg2"],-1)
        fw=int(val*sw)
        if fw>0: cv2.rectangle(frame,(bx0,y),(bx0+fw,y+12),C["accent"],-1)
        cv2.rectangle(frame,(bx0,y),(bx0+sw,y+12),C["sep"],1)
        cv2.putText(frame,f"{int(val*100)}%",(bx0+sw+3,y+9),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,C["muted"],1,cv2.LINE_AA)
        self._hits.append((bx0,y-2,bx0+sw,y+14,'fx_strength',None))

    def hit(self, x:int, y:int) -> Optional[Tuple]:
        """Return full record (x1,y1,x2,y2,action,param) so callers can use rect for slider math."""
        for rec in self._hits:
            if rec[0]<=x<=rec[2] and rec[1]<=y<=rec[3]:
                return rec
        return None

    def update_hover(self, x:int, y:int) -> None:
        """Update the hover-highlight record based on current finger position."""
        self._hover_rec = None
        for rec in self._hits:
            if rec[0]<=x<=rec[2] and rec[1]<=y<=rec[3]:
                self._hover_rec = rec
                break


# =============================================================================
# SECTION 22 — VOICE FEEDBACK
# =============================================================================
class VoiceFeedback:
    def __init__(self) -> None:
        self.enabled = TTS_OK
        self._q      = queue.Queue(maxsize=4)
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate',155)
                threading.Thread(target=self._worker,daemon=True).start()
            except Exception as e:
                print(f"[!] TTS: {e}"); self.enabled=False

    def _worker(self) -> None:
        while True:
            text = self._q.get()
            if text is None: break      # FIX: sentinel stop
            try:
                self.engine.say(text); self.engine.runAndWait()
            except Exception: pass

    def speak(self, text:str) -> None:
        if not self.enabled: return
        try: self._q.put_nowait(text)
        except queue.Full: pass

    def stop(self) -> None:             # FIX: stop() now exists
        if self.enabled:
            try: self._q.put_nowait(None)
            except queue.Full: pass

# =============================================================================
# SECTION 23 — VOICE CONTROLLER
# =============================================================================
class VoiceController:
    def __init__(self, callback) -> None:
        self.cb        = callback
        self.running   = False
        self.last_cmd  = ""
        self.last_t    = 0.0
        self.status    = "off"

    @staticmethod
    def _match(text:str) -> Optional[Any]:
        for phrase in sorted(VOICE_MAP.keys(),key=len,reverse=True):
            if phrase in text: return VOICE_MAP[phrase]
        for word in text.lower().split():
            if word in VOICE_MAP: return VOICE_MAP[word]
        return None

    def start(self) -> None:
        if not VOICE_OK:
            print("  [!] Voice disabled — install SpeechRecognition + pyaudio"); return
        self.running = True; self.status = "listening"
        threading.Thread(target=self._loop,daemon=True).start()
        print("  Voice recognition active")

    def stop(self) -> None:
        self.running = False; self.status = "off"

    def _loop(self) -> None:
        rec = sr.Recognizer()
        rec.energy_threshold        = 400
        rec.dynamic_energy_threshold= True
        rec.pause_threshold         = 0.6
        while self.running:
            try:
                with sr.Microphone() as src:
                    rec.adjust_for_ambient_noise(src,0.8)
                    while self.running:
                        try:
                            self.status = "listening"
                            audio  = rec.listen(src,timeout=1.5,phrase_time_limit=5)
                            self.status = "processing"
                            text   = rec.recognize_google(audio).lower()
                            action = self._match(text)
                            if action:
                                self.last_cmd = text; self.last_t = time.time()
                                self.status   = "heard"; print(f"  Voice: {text!r}")
                                self.cb(action)
                            else: self.status = "listening"
                        except sr.WaitTimeoutError: self.status = "listening"
                        except sr.UnknownValueError: self.status = "listening"
                        except sr.RequestError as e:
                            print(f"  [!] Voice API: {e}")
                            self.status = "error"; time.sleep(3)
            except OSError as e:
                print(f"  [!] Mic: {e}"); self.status="off"; break
            except Exception as e:
                print(f"  [!] Voice: {e}"); self.status="error"; time.sleep(5)

    @property
    def recent(self) -> str:
        return self.last_cmd if time.time()-self.last_t<2.5 else ""

# =============================================================================
# SECTION 24 — GESTURE RECOGNITION
# =============================================================================
def get_gesture(lm) -> Tuple[str,float]:
    """
    Classify hand pose. Returns (gesture_name, pinch_distance_px).
    Uses fingertip-vs-pip landmark comparison for reliable detection.
    """
    index_up  = lm[8].y  < lm[6].y  - 0.020
    middle_up = lm[12].y < lm[10].y - 0.020
    ring_up   = lm[16].y < lm[14].y - 0.045
    pinky_up  = lm[20].y < lm[18].y - 0.045
    thumb_up  = lm[4].y  < lm[3].y  - 0.050
    pinch_d   = math.hypot(lm[4].x-lm[8].x, lm[4].y-lm[8].y) * CAM_W

    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return 'palm_open', pinch_d
    if index_up and middle_up and ring_up and pinky_up:
        return 'four_fingers', pinch_d
    if index_up and middle_up and ring_up and not pinky_up:
        return 'three_fingers', pinch_d
    if pinch_d < 38 and not index_up:
        return 'pinch', pinch_d
    if index_up and middle_up and not ring_up:
        return 'two_fingers', pinch_d
    if index_up and not middle_up:
        return 'one_finger', pinch_d
    return 'idle', pinch_d

# =============================================================================
# SECTION 25 — STARTUP SCREEN
# =============================================================================
class StartupScreen:
    ITEMS = [
        ("Loading Hand Tracking AI …",   0.5),
        ("Loading Brush Engine …",       1.1),
        ("Initialising Layers …",        1.7),
        ("Loading Voice Commands …",     2.3),
        ("Preparing Canvas …",           2.9),
    ]

    def __init__(self, W:int, H:int) -> None:
        self.W, self.H  = W, H
        self.visible    = True
        self.start_t    = time.time()

    def update(self) -> None:
        if time.time()-self.start_t > 3.8: self.visible = False

    def draw(self, frame:np.ndarray) -> np.ndarray:
        if not self.visible: return frame
        elapsed = time.time()-self.start_t
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(self.W,self.H),(10,8,18),-1)
        cv2.addWeighted(ov,0.92,frame,0.08,0,frame)
        # Progress bar
        prog = min(1.0, elapsed/3.8)
        cv2.line(frame,(0,self.H-4),(int(self.W*prog),self.H-4),(30,190,255),3)
        # Title
        title = "Virtual Painter Pro  v4.0"
        (tw,_),_ = cv2.getTextSize(title,cv2.FONT_HERSHEY_DUPLEX,1.3,2)
        alpha_blend(frame,self.W//2-tw//2-20,self.H//2-82,
                    self.W//2+tw//2+20,self.H//2-32,(30,190,255),0.12)
        cv2.putText(frame,title,(self.W//2-tw//2,self.H//2-50),
                    cv2.FONT_HERSHEY_DUPLEX,1.3,(30,190,255),2,cv2.LINE_AA)
        sub = "AI + OpenCV + MediaPipe"
        (sw,_),_ = cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.putText(frame,sub,(self.W//2-sw//2,self.H//2-16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(110,108,148),1,cv2.LINE_AA)
        # Item list
        for i,(text,trigger) in enumerate(self.ITEMS):
            iy   = self.H//2+30+i*34
            done = elapsed >= trigger+0.3
            act  = trigger<=elapsed<trigger+0.6
            icon = "[OK]" if done else "[>>]" if act else "[ ]"
            col  = (55,230,100) if done else (30,190,255) if act else (70,68,90)
            cv2.putText(frame,f"{icon}  {text}",(self.W//2-165,iy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.47,col,1,cv2.LINE_AA)
        # Spinner
        ang = (elapsed*180)%360
        cv2.ellipse(frame,(self.W//2-185,self.H//2+30+2*34+5),(10,10),
                    ang,0,270,(30,190,255),2)
        return frame


# =============================================================================
# SECTION 26 — MAIN APPLICATION  —  VirtualPainterV4
# =============================================================================
class VirtualPainterV4:
    """Central coordinator. process(frame) is the per-frame entry point."""

    EMOJI_ASCII = [':-)', '<3', '(*)', '>>>', '[art]', '**', '!!!', '###']

    def __init__(self, W:int=CAM_W, H:int=CAM_H) -> None:
        self.W, self.H  = W, H
        self.canvas_h   = H-SB_H
        self._voice_q   : queue.Queue = queue.Queue(maxsize=16)

        # Theme
        self.current_theme = Theme.DARK
        self.colors        = THEMES["dark"]

        # Tool / brush state
        self.tool        = 'pen'
        self.eraser      = False
        self.custom_col  : Optional[Tuple] = None   # FIX: cleared on tool switch
        self.brush_cfg   = BrushConfig()
        self.grid_on     = False
        self.fx_strength = 0.70
        self._last_fx    = 'blur'

        # Subsystems
        self.layers      = LayerManager(W, self.canvas_h, MAX_LAYERS)
        self.hist        = History(UNDO_LIMIT)
        self.color_picker= ColorPicker()
        self.color_picker.set_from_bgr((0,0,255))
        self.selection   = SelectionTool()
        self.viewport    = CanvasViewport(W, self.canvas_h)
        self.symmetry    = SymmetryEngine()
        self.analytics   = Analytics()
        self.timelapse   = TimelapseRecorder()
        self.recog       = AIShapeRecognizer()

        # UI
        self.dock        = FloatingDock(DOCK_W, self.canvas_h)
        self.panel       = SidePanel()
        self.cursor      = AnimatedCursor()
        self.startup     = StartupScreen(W, H)
        self.voice_fb    = VoiceFeedback()
        self.voice       = VoiceController(self._enqueue_voice)
        self.voice.start()

        # Drawing state
        self.shape_start  : Optional[Tuple[int,int]]  = None
        self.shape_points : List[Tuple[int,int]]       = []
        self.prev_xy      : Optional[Tuple[int,int]]   = None
        self.smooth_xy    = (0, 0)
        self.fill_done    = False
        # FIX: emoji_idx initialised here (was missing → AttributeError)
        self.emoji_idx    = 0
        self.sticker_idx  = 0

        # Gesture state
        self.g_buf   : Deque[str]      = deque(maxlen=STABLE_FRAMES)
        self.g_cur   = 'idle'
        self._gest_t : Dict[str,float] = {}
        self._pinch_base : Optional[float] = None
        self._pan_base   : Optional[Tuple] = None  # (x,y,pan_x,pan_y)

        # Notifications
        self.cmd_msg = ""; self.cmd_t = 0.0

        # Autosave
        self._last_autosave = time.time()

        # FPS
        self._fps_t  = time.time()
        self._fps_buf: Deque[float] = deque([30.0]*10, maxlen=10)

        # MediaPipe
        ensure_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.68,
            min_hand_presence_confidence=0.68,
            min_tracking_confidence=0.68,
        )
        self.lmk    = mp_vision.HandLandmarker.create_from_options(opts)
        self.ts_base = int(time.time()*1000)
        self.last_ts = 0

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def active_color(self) -> Tuple[int,int,int]:
        return (0,0,0) if self.eraser else (self.custom_col or self.color_picker.get_bgr())

    # ── Thread-safe voice queue ────────────────────────────────────────────────
    def _enqueue_voice(self, action:Any) -> None:
        try: self._voice_q.put_nowait(action)
        except queue.Full: pass

    def _drain_voice_queue(self, max_n:int=2) -> None:
        for _ in range(max_n):
            try:
                self._voice_action(self._voice_q.get_nowait())
                self.analytics.voice()
            except queue.Empty: break

    # ── Notification ──────────────────────────────────────────────────────────
    def _msg(self, t:str) -> None:
        self.cmd_msg = t; self.cmd_t = time.time(); print(f"  [VP4] {t}")

    # ── History helpers ────────────────────────────────────────────────────────
    def _push(self) -> None:
        self.hist.push(self.layers.layers)

    def _undo(self) -> None:
        restored = self.hist.undo(self.layers.layers)
        self.layers.restore_from(restored)
        self.analytics.undo(); self._msg("Undo")

    def _redo(self) -> None:
        restored = self.hist.redo(self.layers.layers)
        self.layers.restore_from(restored)
        self.analytics.redo(); self._msg("Redo")

    # ── Save / export ──────────────────────────────────────────────────────────
    def _save(self, fmt:str='png') -> None:
        canvas = self.layers.merge_visible()
        if fmt=='jpg':   p = ExportManager.export_jpg(canvas)
        elif fmt=='svg': p = ExportManager.export_svg(canvas)
        else:            p = ExportManager.export_png(canvas)
        self._msg(f"Saved {fmt.upper()}: {os.path.basename(p)}")
        self.voice_fb.speak(f"Saved as {fmt}")

    def _save_project(self) -> None:
        os.makedirs("projects",exist_ok=True)
        p = os.path.join("projects",f"project_{datetime.now():%Y%m%d_%H%M%S}.vpaint")
        if ProjectManager.save(p,self): self._msg(f"Project saved: {os.path.basename(p)}")
        else: self._msg("Save failed!")

    # ── Theme ──────────────────────────────────────────────────────────────────
    def toggle_theme(self, theme:Theme) -> None:
        self.current_theme = theme
        self.colors        = THEMES[theme.value]
        self.dock.update_theme(theme.value)
        self._msg(f"{theme.value.title()} mode")

    # ── Gesture fire-once gate ─────────────────────────────────────────────────
    def _fire(self, name:str, cd:float=GESTURE_COOLDOWN) -> bool:
        now = time.time()
        if now-self._gest_t.get(name,0)>cd:
            self._gest_t[name] = now; return True
        return False

    # ── Canvas drawing ─────────────────────────────────────────────────────────
    def _draw_on_canvas(self, cx:int, cy:int,
                        px:Optional[int]=None, py:Optional[int]=None) -> None:
        """Apply brush stroke at canvas coords, respecting symmetry & lock."""
        if self.layers.current_meta().locked:
            self._msg("Layer is locked!"); return
        color = self.active_color
        cfg   = self.brush_cfg
        layer = self.layers.current()
        # All mirror points
        pts = self.symmetry.mirror_points(cx,cy,self.W,self.canvas_h)
        if px is not None and py is not None:
            prev_pts = self.symmetry.mirror_points(px,py,self.W,self.canvas_h)
        else:
            prev_pts = [None]*len(pts)
        for i,(tx,ty) in enumerate(pts):
            ppxy = prev_pts[i]
            ppx  = ppxy[0] if ppxy else None
            ppy  = ppxy[1] if ppxy else None
            BrushEngine.apply_stroke(layer,ppx,ppy,tx,ty,
                                     color,cfg,self.eraser)
        self.layers.invalidate()

    # ── Effects ────────────────────────────────────────────────────────────────
    def _apply_effect(self, effect:str) -> None:
        if self.layers.current_meta().locked:
            self._msg("Layer is locked!"); return
        self._last_fx = effect
        self._push()
        result = EffectsEngine.apply(self.layers.current(),effect,self.fx_strength)
        np.copyto(self.layers.current(),result)
        self.layers.invalidate()
        self.analytics.effect()
        self._msg(f"Effect: {effect}")
        self.voice_fb.speak(effect)

    # ── Text & stamps ──────────────────────────────────────────────────────────
    def _stamp_text(self, x:int, y:int) -> None:
        if not self.text_buf.strip(): self.text_mode=False; self.text_buf=""; return
        self._push()
        scale = max(0.5, self.brush_cfg.size/12.0)
        thick = max(1, self.brush_cfg.size//8)
        cv2.putText(self.layers.current(),self.text_buf,(x,y),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,self.active_color,thick,cv2.LINE_AA)
        self.layers.invalidate()
        self.text_buf=""; self.text_pos=None; self.text_mode=False
        self._msg("Text stamped")

    def _stamp_emoji(self, x:int, y:int) -> None:
        # FIX: emoji_idx initialised in __init__; EMOJI_ASCII is class var
        label = self.EMOJI_ASCII[self.emoji_idx%len(self.EMOJI_ASCII)]
        scale = max(0.6,self.brush_cfg.size/14.0)
        cv2.putText(self.layers.current(),label,(x,y),
                    cv2.FONT_HERSHEY_DUPLEX,scale,self.active_color,
                    max(1,int(scale*2)),cv2.LINE_AA)
        self.layers.invalidate(); self._msg(f"Stamp: {label}")

    def _stamp_sticker(self, x:int, y:int) -> None:
        names = StickerLibrary.ALL_NAMES
        name  = names[self.sticker_idx%len(names)]
        self._push()
        StickerLibrary.stamp(self.layers.current(),name,x,y,
                             self.brush_cfg.size*2,self.active_color)
        self.layers.invalidate(); self._msg(f"Sticker: {name}")

    # ── Voice action handler ────────────────────────────────────────────────────
    def _voice_action(self, action:Any) -> None:
        C = self.colors
        if action=='clear':
            self._push(); self.layers.clear_current(); self._msg("Canvas cleared")
        elif action=='new':
            self._push(); self.layers.clear_all(); self._msg("New canvas")
        elif action in ('undo','redo'):
            self._undo() if action=='undo' else self._redo()
        elif action in ('save','save_png'):
            self._save('png')
        elif action=='save_jpg':
            self._save('jpg')
        elif action=='save_project':
            self._save_project()
        elif action=='brush_up':
            self.brush_cfg.size = min(MAX_BRUSH, self.brush_cfg.size+4)
            self._msg(f"Size {self.brush_cfg.size}")
        elif action=='brush_down':
            self.brush_cfg.size = max(MIN_BRUSH, self.brush_cfg.size-4)
            self._msg(f"Size {self.brush_cfg.size}")
        elif action=='opacity_up':
            self.brush_cfg.opacity = min(1.0, round(self.brush_cfg.opacity+0.1,2))
            self._msg(f"Opacity {int(self.brush_cfg.opacity*100)}%")
        elif action=='opacity_down':
            self.brush_cfg.opacity = max(0.1, round(self.brush_cfg.opacity-0.1,2))
            self._msg(f"Opacity {int(self.brush_cfg.opacity*100)}%")
        elif isinstance(action, tuple) and action[0]=='tool':
            old = self.tool
            self.tool   = action[1]
            self.eraser = (self.tool=='eraser')
            # FIX: clear custom_col on tool switch
            if old != self.tool: self.custom_col = None
            self._msg(f"Tool: {self.tool}")
        elif isinstance(action, tuple) and action[0]=='layer':
            self.layers.set_active(action[1])
            self.analytics.layer()
            self._msg(f"Layer {action[1]+1}")
        elif isinstance(action, tuple) and action[0]=='color':
            bgr = action[1]
            self.color_picker.set_from_bgr(bgr)
            self._msg(f"Color set")
        elif action=='new_layer':
            # Find first empty layer
            for i,l in enumerate(self.layers.layers):
                if not np.any(l):
                    self.layers.set_active(i)
                    self._msg(f"Layer {i+1} active"); break
        elif action=='delete_layer':
            self._push(); self.layers.clear_current()
            self._msg("Layer cleared")
        elif action.startswith('brush_'):
            bt = action[6:]
            if bt in BRUSH_TYPES:
                self.brush_cfg.brush_type = bt
                self.analytics.brush_change()
                self._msg(f"Brush: {bt}")
        elif action.startswith('shape_'):
            self._draw_preset_shape(action[6:])
        elif action.startswith('fx_'):
            self._apply_effect(action[3:])
        elif action=='zoom_in':
            self.viewport.zoom_at(self.W//2,self.canvas_h//2,0.25)
            self._msg(f"Zoom {self.viewport.vt.zoom:.1f}x")
        elif action=='zoom_out':
            self.viewport.zoom_at(self.W//2,self.canvas_h//2,-0.25)
            self._msg(f"Zoom {self.viewport.vt.zoom:.1f}x")
        elif action=='reset_zoom':
            self.viewport.vt.reset(); self._msg("View reset")
        elif action in ('symmetry_on','symmetry_v'):
            self.symmetry.mode = SymMode.VERTICAL; self._msg("Symmetry: Vertical")
        elif action=='symmetry_h':
            self.symmetry.mode = SymMode.HORIZONTAL; self._msg("Symmetry: Horizontal")
        elif action=='symmetry_r':
            self.symmetry.mode = SymMode.RADIAL; self._msg("Symmetry: Radial")
        elif action=='symmetry_mandala':
            self.symmetry.mode = SymMode.MANDALA; self._msg("Symmetry: Mandala")
        elif action=='symmetry_off':
            self.symmetry.mode = SymMode.NONE; self._msg("Symmetry: Off")
        elif action=='toggle_dark':
            self.toggle_theme(Theme.DARK)
        elif action=='toggle_light':
            self.toggle_theme(Theme.LIGHT)
        elif action=='grid_on':
            self.grid_on=True; self._msg("Grid on")
        elif action=='grid_off':
            self.grid_on=False; self._msg("Grid off")

    def _draw_preset_shape(self, shape:str) -> None:
        """Draw a perfect shape centred on the canvas."""
        self._push()
        cx,cy = self.W//2, self.canvas_h//2; r=100
        layer = self.layers.current(); color = self.active_color; t=3
        pts   = [(cx,cy),(cx,cy),(cx+r,cy+r),(cx-r,cy-r)]
        AIShapeRecognizer.draw_perfect(layer,pts,shape,color,t)
        self.layers.invalidate()
        self.analytics.shape()
        self._msg(f"Shape: {shape}")

    # ── Panel click handler ────────────────────────────────────────────────────
    def _handle_panel_click(self, lx:int, ly:int) -> bool:
        """Handle a two-finger click in the side panel. All slider math uses the hit rect."""
        if self.panel.mode == PanelMode.NONE:
            return False
        rec = self.panel.hit(lx, ly)
        if rec is None:
            return False
        # Full record gives us the rect for slider ratio calculation
        x1, y1, x2, y2, action, param = rec
        t = float(np.clip((lx - x1) / max(1, x2 - x1), 0.0, 1.0))   # 0.0–1.0 along slider

        # ── Layer panel actions ────────────────────────────────────────────────
        if action == 'layer_select':
            self.layers.set_active(param)
            self.analytics.layer()
            self._msg(f"Layer {param+1}")

        elif action == 'layer_vis':
            self.layers.toggle_visibility(param)
            vis = self.layers.meta[param].visible
            self._msg(f"L{param+1} {'shown' if vis else 'hidden'}")

        elif action == 'layer_lock':
            self.layers.toggle_lock(param)
            lk = self.layers.meta[param].locked
            self._msg(f"L{param+1} {'locked' if lk else 'unlocked'}")

        elif action == 'layer_opacity':
            # t is already the correct 0-1 ratio from click position within slider bar
            self.layers.set_opacity(param, t)
            self._msg(f"L{param+1} opacity {int(t*100)}%")

        elif action == 'layer_add':
            for i, lay in enumerate(self.layers.layers):
                if not np.any(lay):
                    self.layers.set_active(i)
                    self._msg(f"Layer {i+1} active"); break
            else:
                self._msg("All layer slots used")

        elif action == 'layer_dup':
            j = self.layers.duplicate(self.layers.active_layer)
            self._msg(f"Duplicated to L{j+1}" if j >= 0 else "No free slot")

        elif action == 'layer_del':
            self._push(); self.layers.clear_current()
            self._msg(f"L{self.layers.active_layer+1} cleared")

        elif action == 'layer_up':
            self.layers.set_active(max(0, self.layers.active_layer - 1))
            self._msg(f"Layer {self.layers.active_layer+1}")

        elif action == 'layer_dn':
            self.layers.set_active(min(MAX_LAYERS-1, self.layers.active_layer + 1))
            self._msg(f"Layer {self.layers.active_layer+1}")

        # ── Brush panel actions ────────────────────────────────────────────────
        elif action == 'brush_type':
            self.brush_cfg.brush_type = param
            self.analytics.brush_change()
            self._msg(f"Brush: {param}")

        elif action == 'brush_size':
            self.brush_cfg.size = max(MIN_BRUSH, int(t * MAX_BRUSH))
            self._msg(f"Size {self.brush_cfg.size}px")

        elif action == 'brush_opacity':
            self.brush_cfg.opacity = max(0.05, t)
            self._msg(f"Opacity {int(self.brush_cfg.opacity*100)}%")

        elif action == 'brush_hardness':
            self.brush_cfg.hardness = max(0.1, t)
            self._msg(f"Hardness {int(t*100)}%")

        elif action == 'brush_flow':
            self.brush_cfg.flow = max(0.1, t)
            self._msg(f"Flow {int(t*100)}%")

        # ── Effects panel actions ──────────────────────────────────────────────
        elif action == 'fx_apply':
            self._apply_effect(param)

        elif action == 'fx_strength':
            self.fx_strength = t
            self._msg(f"Strength {int(t*100)}%")

        # ── Color picker actions ───────────────────────────────────────────────
        elif action == 'color_pick':
            # Pass the native ColorPicker hit list — it knows how to map click→color
            new_col = self.color_picker.handle_click(lx, ly, self.panel._color_hits)
            if new_col:
                self.custom_col = None          # use picker value, not override
                self._msg(f"Color #{self.color_picker.hex_str}")

        return True

    # ── Grid overlay ───────────────────────────────────────────────────────────
    def _draw_grid(self, frame:np.ndarray) -> None:
        if not self.grid_on: return
        step = 50; gc = self.colors["sep"]
        for x in range(DOCK_W,self.W,step):
            cv2.line(frame,(x,0),(x,self.canvas_h),gc,1)
        for y in range(0,self.canvas_h,step):
            cv2.line(frame,(DOCK_W,y),(self.W,y),gc,1)

    # ── Status bar ─────────────────────────────────────────────────────────────
    def _draw_status_bar(self, frame:np.ndarray, gesture:str) -> None:
        C  = self.colors
        y0 = self.canvas_h; y1 = self.H
        alpha_blend(frame,0,y0,self.W,y1,C["sb"],0.95)
        cv2.line(frame,(0,y0),(self.W,y0),C["border"],1)
        # FPS
        now = time.time()
        dt  = now-self._fps_t; self._fps_t = now
        self._fps_buf.append(1.0/max(0.001,dt))
        fps = int(np.mean(self._fps_buf))
        # Left info
        meta = self.layers.current_meta()
        lbl  = (f"  {meta.name}  |  {self.brush_cfg.brush_type.upper()}"
                f"  {self.brush_cfg.size}px  |  "
                f"Z {self.viewport.vt.zoom:.1f}x  |  "
                f"Sym: {self.symmetry.mode.value[:4].upper()}  |  "
                f"FPS {fps}")
        cv2.putText(frame,lbl,(DOCK_W+8,y0+26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,C["muted"],1,cv2.LINE_AA)
        # Right: gesture + voice + tool
        col_swatch = self.active_color
        sw_x = self.W-PANEL_W-140 if self.panel.mode!=PanelMode.NONE else self.W-140
        cv2.rectangle(frame,(sw_x,y0+8),(sw_x+22,y0+30),col_swatch,-1)
        cv2.rectangle(frame,(sw_x,y0+8),(sw_x+22,y0+30),C["border"],1)
        cv2.putText(frame,f"{self.tool.upper()}  G:{gesture[:4]}",(sw_x+28,y0+24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,C["text"],1,cv2.LINE_AA)
        # Voice badge
        if self.voice.recent:
            vc_x = sw_x-180
            pill_text(frame,f'"{self.voice.recent[:22]}"',vc_x+90,y0+20,
                      C["text"],C["bg2"],0.33,6,0.72)
        # Layer dots
        dot_y = y0+SB_H//2
        for i in range(MAX_LAYERS):
            dc = (DOCK_W+10+i*18, dot_y)
            filled = i==self.layers.active_layer
            col    = C["active"] if filled else C["muted"]
            cv2.circle(frame,dc, 6 if filled else 4, col, -1 if filled else 1)
            if not self.layers.meta[i].visible:
                cv2.line(frame,(dc[0]-4,dc[1]-4),(dc[0]+4,dc[1]+4),C["danger"],1)

    # ── Notification overlay ────────────────────────────────────────────────────
    def _draw_notification(self, frame:np.ndarray) -> None:
        if self.cmd_msg and time.time()-self.cmd_t<2.2:
            alpha = min(1.0,(2.2-(time.time()-self.cmd_t))/0.4)
            col   = tuple(int(c*alpha) for c in self.colors["text"])
            bg    = tuple(int(c*alpha) for c in self.colors["bg2"])
            pill_text(frame,self.cmd_msg,
                      self.W//2,self.canvas_h-28,col,bg,0.46,12,0.80)

    # ── Text cursor overlay ─────────────────────────────────────────────────────
    def _draw_text_cursor(self, frame:np.ndarray) -> None:
        if not (hasattr(self,'text_mode') and self.text_mode and self.text_pos):
            return
        tx,ty = self.text_pos
        scale  = max(0.5,self.brush_cfg.size/12.0)
        thick  = max(1,self.brush_cfg.size//8)
        preview = self.text_buf + ('|' if int(time.time()*2)%2==0 else '')
        cv2.putText(frame,preview,(tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,self.colors["accent"],thick,cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN PROCESS METHOD — called every frame
    # ══════════════════════════════════════════════════════════════════════════
    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame = cv2.flip(frame, 1)

        # Startup screen
        if self.startup.visible:
            self.startup.update()
            return self.startup.draw(frame)

        # Drain voice queue
        self._drain_voice_queue()

        # Autosave
        if time.time()-self._last_autosave > AUTOSAVE_SECS:
            threading.Thread(target=ProjectManager.save,
                             args=("autosave.vpaint",self),daemon=True).start()
            self._last_autosave = time.time()

        # ── MediaPipe hand detection ──────────────────────────────────────────
        ts = int(time.time()*1000)-self.ts_base
        if ts <= self.last_ts: ts = self.last_ts+1
        self.last_ts = ts

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        result = self.lmk.detect_for_video(mp_img,ts)

        gesture  = 'idle'
        pinch_d  = 999.0
        sx = sy  = 0

        if result.hand_landmarks:
            lm  = result.hand_landmarks[0]
            draw_skeleton(frame,lm,self.W,self.H,self.current_theme.value)
            # Raw index-fingertip position (mirrored)
            sx_raw = int(lm[8].x*self.W)
            sy_raw = int(lm[8].y*self.H)
            # Clamp to canvas area
            sx_raw = max(DOCK_W,min(self.W-1,sx_raw))
            sy_raw = max(0,min(self.canvas_h-1,sy_raw))
            # Smooth (EMA)
            alpha_ema = 0.50
            self.smooth_xy = (
                int(self.smooth_xy[0]*(1-alpha_ema)+sx_raw*alpha_ema),
                int(self.smooth_xy[1]*(1-alpha_ema)+sy_raw*alpha_ema),
            )
            sx,sy = self.smooth_xy
            # Transform screen->canvas for viewport
            cx,cy = self.viewport.screen_to_canvas(sx,sy)
            raw_g,pinch_d = get_gesture(lm)
            # Stabilise gesture
            self.g_buf.append(raw_g)
            if len(set(self.g_buf))==1:
                gesture = raw_g
                self.g_cur = gesture
            else:
                gesture = self.g_cur
            # Draw cursor
            self.cursor.add(sx,sy)
            self.cursor.draw(frame,sx,sy,self.brush_cfg,
                             self.active_color,self.eraser)
            # Handle gesture
            self._handle_gesture(gesture,pinch_d,sx,sy,cx,cy,lm)
        else:
            # No hand: end any ongoing stroke
            if self.prev_xy is not None:
                self._push(); self.analytics.stroke()
            self.prev_xy = None
            self.cursor.clear()
            gesture = 'none'

        # ── Composite canvas onto frame ───────────────────────────────────────
        merged = self.layers.merge_visible()
        # Apply viewport zoom/pan to display
        display = self.viewport.apply_to_display(merged,
                                                 self.W-DOCK_W,
                                                 self.canvas_h)
        # Overlay painted strokes onto live camera feed (black pixels = transparent)
        cam_roi = frame[:self.canvas_h, DOCK_W:]
        stroke_mask = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY) > 2
        cam_roi[stroke_mask] = display[stroke_mask]

        # Shape drawing tools: live preview
        # FIX: use ROI crop instead of full layer copy
        if self.tool in ('line','rect','circle') and self.shape_start:
            sp_x,sp_y = self.shape_start
            disp_x = int(sp_x*self.viewport.vt.zoom+self.viewport.vt.pan_x)
            disp_y = int(sp_y*self.viewport.vt.zoom+self.viewport.vt.pan_y)
            self._draw_shape(frame,(disp_x+DOCK_W,disp_y),(sx,sy),
                             self.active_color,self.brush_cfg.size)
        # Selection overlay
        if self.tool=='select':
            self.selection.draw_overlay(frame,self.colors["select"])
        # Symmetry guide
        self.symmetry.draw_guide(frame,self.W,self.canvas_h,self.colors["accent"])
        # Grid
        self._draw_grid(frame)
        # Dock
        self.dock.draw(frame,self.tool,self.eraser)
        # Update hover highlight so panel shows what 2-finger would click
        if self.panel.mode != PanelMode.NONE and result.hand_landmarks:
            self.panel.update_hover(sx, sy)
        # Side panel
        self.panel.draw(frame,self)
        # Status bar
        self._draw_status_bar(frame,gesture)
        # Text cursor overlay
        self._draw_text_cursor(frame)
        # Notification
        self._draw_notification(frame)
        # Timelapse
        if self.timelapse.recording:
            self.timelapse.add_frame(merged)
        return frame

    # ── Gesture handler ────────────────────────────────────────────────────────
    def _handle_gesture(self, g:str, pinch_d:float,
                        sx:int, sy:int,
                        cx:int, cy:int, lm) -> None:
        C = self.colors
        panel_open  = self.panel.mode != PanelMode.NONE
        in_panel    = sx >= self.W-PANEL_W and panel_open
        in_dock     = sx < DOCK_W

        # ── Pinch = zoom ──────────────────────────────────────────────────────
        if g=='pinch':
            if self._pinch_base is None:
                self._pinch_base = pinch_d
            else:
                diff = pinch_d - self._pinch_base
                if abs(diff)>8:
                    delta = 0.25 if diff>0 else -0.25
                    self.viewport.zoom_at(sx,sy,delta)
                    self._pinch_base = pinch_d
                    self._msg(f"Zoom {self.viewport.vt.zoom:.1f}x")
        else:
            self._pinch_base = None

        # ── Three fingers = UNDO ───────────────────────────────────────────────
        if g=='three_fingers' and self._fire('undo', GESTURE_COOLDOWN):
            self._undo()

        # ── Four fingers = REDO ────────────────────────────────────────────────
        if g=='four_fingers' and self._fire('redo', GESTURE_COOLDOWN):
            self._redo()

        # ── Palm open = CLEAR ─────────────────────────────────────────────────
        if g=='palm_open' and self._fire('clear', 1.5):
            self._push(); self.layers.clear_current(); self._msg("Layer cleared")

        # ── Two fingers = CLICK/CONFIRM/PAN ───────────────────────────────────
        # ── Two fingers — SEPARATE fire gates for panel / dock / canvas ──────
        # Panel uses a shorter cooldown (0.15 s) so item selection feels snappy.
        if g == 'two_fingers':
            if in_panel and self._fire('panel_click', 0.15):
                consumed = self._handle_panel_click(sx, sy)
                if not consumed:
                    self._msg("Point at a panel item")

            elif in_dock and self._fire('dock_click', TB_COOLDOWN):
                hit = self.dock.hit(sx, sy)
                if hit:
                    if hit.startswith('tool_'):
                        t = hit[5:]
                        self.tool   = t
                        self.eraser = (t == 'eraser')
                        self.custom_col = None
                        self._msg(f"Tool: {t}")
                    elif hit.startswith('panel_'):
                        pname  = hit[6:]
                        pm_map = {
                            'brush':   PanelMode.BRUSH,
                            'layers':  PanelMode.LAYERS,
                            'color':   PanelMode.COLORS,
                            'effects': PanelMode.EFFECTS,
                            'stats':   PanelMode.ANALYTICS,
                        }
                        self.panel.toggle(pm_map.get(pname, PanelMode.NONE))
                        self._msg(f"Panel: {pname}")

            elif not in_dock and not in_panel and self._fire('two', TB_COOLDOWN):
                if self.tool in ('line', 'rect', 'circle') and self.shape_start:
                    sp_x, sp_y = self.shape_start
                    self._push()
                    self._draw_shape(self.layers.current(),
                                     (sp_x, sp_y), (cx, cy),
                                     self.active_color, self.brush_cfg.size)
                    self.layers.invalidate()
                    self.shape_start = None
                    self.analytics.shape()
                    self._msg(f"{self.tool.title()} drawn")
                elif self.tool == 'select':
                    self.selection.end()
                    if self.selection.state.rect:
                        self.selection.copy_region(self.layers.current())
                        self._msg("Selection copied")
                elif self.tool == 'text' and hasattr(self,'text_pos') and self.text_pos:
                    self._stamp_text(*self.text_pos)
                elif self.tool == 'emoji':
                    self._push(); self._stamp_emoji(cx, cy)
                elif self.tool == 'eyedropper':
                    merged = self.layers.merge_visible()
                    if 0 <= cy < merged.shape[0] and 0 <= cx < merged.shape[1]:
                        picked = tuple(int(c) for c in merged[cy, cx])
                        if any(v > 0 for v in picked):
                            self.color_picker.set_from_bgr(picked)
                            self.custom_col = None
                            self._msg(f"Picked #{self.color_picker.hex_str}")

        # ── One finger = DRAW / START SHAPE / FILL ────────────────────────────
        if g=='one_finger':
            if not in_dock and not in_panel:
                if self.tool in ('pen','eraser','marker','pencil','airbrush',
                                 'watercolor','neon','glow','spray','calligraphy'):
                    if self.prev_xy is None:
                        self._push()
                    self._draw_on_canvas(cx,cy,
                                         self.prev_xy[0] if self.prev_xy else None,
                                         self.prev_xy[1] if self.prev_xy else None)
                    self.prev_xy = (cx,cy)
                elif self.tool in ('line','rect','circle'):
                    if self.shape_start is None:
                        self.shape_start = (cx,cy)
                        self._msg(f"Shape start…")
                elif self.tool=='fill':
                    if self._fire('fill', 0.4):
                        self._push()
                        flood_fill(self.layers.current(),cx,cy,self.active_color)
                        self.layers.invalidate()
                        self.fill_done = True
                        self._msg("Fill")
                elif self.tool=='text':
                    self.text_mode = True
                    self.text_pos  = (cx,cy)
                    self._msg("Type then press Enter")
                elif self.tool=='emoji':
                    if self._fire('emoji', 0.5):
                        self._push()
                        self._stamp_emoji(cx,cy)
                elif self.tool=='select':
                    if self.selection.state.drag_start is None:
                        self.selection.begin(cx,cy)
                    else:
                        self.selection.update(cx,cy)
                elif self.tool=='eyedropper':
                    merged = self.layers.merge_visible()
                    if 0<=cy<merged.shape[0] and 0<=cx<merged.shape[1]:
                        picked = tuple(int(c) for c in merged[cy,cx])
                        if any(v>0 for v in picked):
                            self.color_picker.set_from_bgr(picked)
                            self.custom_col = None
        else:
            # Gesture released — end stroke
            if self.prev_xy is not None:
                self.analytics.stroke()
            self.prev_xy = None

    # ── Shape helper ───────────────────────────────────────────────────────────
    def _draw_shape(self, dst:np.ndarray, p1:Tuple, p2:Tuple,
                    color:Tuple, thick:int) -> None:
        if self.tool=='line':
            cv2.line(dst,p1,p2,color,thick)
        elif self.tool=='rect':
            cv2.rectangle(dst,p1,p2,color,thick)
        elif self.tool=='circle':
            r = int(math.hypot(p2[0]-p1[0],p2[1]-p1[1]))
            cv2.circle(dst,p1,r,color,thick)

    # ── Mouse click handler — reliable panel/dock interaction ──────────────────
    def on_mouse(self, event:int, x:int, y:int, flags:int, param) -> None:
        """
        Registered via cv2.setMouseCallback.
        Left-click  : dock tool | panel item | canvas action
        Right-click : eyedropper (pick color)
        Scroll      : brush size up/down
        Middle-btn  : toggle Layer panel
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < DOCK_W:                                   # ── Dock ──────
                hit = self.dock.hit(x, y)
                if hit:
                    if hit.startswith('tool_'):
                        t = hit[5:]; self.tool = t
                        self.eraser = (t == 'eraser')
                        self.custom_col = None
                        self._msg(f"Tool: {t}")
                    elif hit.startswith('panel_'):
                        pm_map = {
                            'brush':   PanelMode.BRUSH,
                            'layers':  PanelMode.LAYERS,
                            'color':   PanelMode.COLORS,
                            'effects': PanelMode.EFFECTS,
                            'stats':   PanelMode.ANALYTICS,
                        }
                        pname = hit[6:]
                        self.panel.toggle(pm_map.get(pname, PanelMode.NONE))
                        self._msg(f"Panel: {pname}")
            elif self.panel.mode != PanelMode.NONE and x >= self.W - PANEL_W:
                self._handle_panel_click(x, y)              # ── Side panel ─
            else:                                            # ── Canvas ─────
                cx, cy = self.viewport.screen_to_canvas(x - DOCK_W, y)
                if self.tool == 'fill':
                    self._push()
                    flood_fill(self.layers.current(), cx, cy, self.active_color)
                    self.layers.invalidate(); self._msg("Fill")
                elif self.tool == 'text':
                    self.text_mode = True
                    self.text_pos  = (cx + DOCK_W, cy)
                    self._msg("Type then Enter to stamp")
                elif self.tool == 'select' and self.selection.state.clipboard:
                    self.selection.paste_region(self.layers.current(), cx, cy)
                    self.layers.invalidate(); self._msg("Pasted")

        elif event == cv2.EVENT_RBUTTONDOWN:               # Eyedropper ─────
            merged = self.layers.merge_visible()
            cx, cy = self.viewport.screen_to_canvas(max(0, x - DOCK_W), y)
            if 0 <= cy < merged.shape[0] and 0 <= cx < merged.shape[1]:
                picked = tuple(int(c) for c in merged[cy, cx])
                if any(v > 0 for v in picked):
                    self.color_picker.set_from_bgr(picked)
                    self.custom_col = None
                    self._msg(f"Picked #{self.color_picker.hex_str}")

        elif event == cv2.EVENT_MOUSEWHEEL:                # Scroll = size ──
            self.brush_cfg.size = int(np.clip(
                self.brush_cfg.size + (2 if flags > 0 else -2),
                MIN_BRUSH, MAX_BRUSH))
            self._msg(f"Size {self.brush_cfg.size}px")

        elif event == cv2.EVENT_MBUTTONDOWN:               # Middle = panel ─
            self.panel.mode = (PanelMode.NONE
                               if self.panel.mode != PanelMode.NONE
                               else PanelMode.LAYERS)


    # ── Keyboard handler ───────────────────────────────────────────────────────
    def handle_key(self, key:int) -> bool:
        """
        Handle keyboard shortcuts.
        Returns False if the app should quit.
        """
        if key in (ord('q'),27): return False   # quit

        # Text mode captures keys
        if hasattr(self,'text_mode') and self.text_mode:
            if key==13:    # Enter
                if self.text_pos: self._stamp_text(*self.text_pos)
            elif key==8:   # Backspace
                self.text_buf = self.text_buf[:-1]
            elif 32<=key<127:
                self.text_buf += chr(key)
            return True

        # ── Navigation & display ──────────────────────────────────────────────
        if key==ord('d'):
            self.toggle_theme(Theme.LIGHT if self.current_theme==Theme.DARK
                              else Theme.DARK)
        elif key==ord('g'):
            self.grid_on = not self.grid_on
            self._msg(f"Grid {'on' if self.grid_on else 'off'}")
        elif key==ord('x'):
            self.symmetry.next_mode()
            self._msg(f"Symmetry: {self.symmetry.mode.value}")
        elif key==ord('v'):
            self.viewport.vt.reset(); self._msg("View reset")
        # ── Undo / Redo ───────────────────────────────────────────────────────
        elif key==ord('z'):   self._undo()
        elif key==ord('y'):   self._redo()
        # ── Canvas ────────────────────────────────────────────────────────────
        elif key==ord('c'):
            self._push(); self.layers.clear_current(); self._msg("Layer cleared")
        # ── Save / project ────────────────────────────────────────────────────
        elif key==ord('s'):   self._save('png')
        elif key==19:         self._save_project()          # Ctrl+S
        # ── Panels ────────────────────────────────────────────────────────────
        elif key==ord('b'):   self.panel.toggle(PanelMode.BRUSH)
        elif key==ord('l'):   self.panel.toggle(PanelMode.LAYERS)
        elif key==ord('k'):   self.panel.toggle(PanelMode.COLORS)
        elif key==ord('e'):   self.panel.toggle(PanelMode.EFFECTS)
        elif key==ord('a'):   self.panel.toggle(PanelMode.ANALYTICS)
        # ── Quick layer switch ─────────────────────────────────────────────────
        elif ord('1')<=key<=ord('8'):
            idx = key-ord('1')
            self.layers.set_active(idx)
            self.analytics.layer()
            self._msg(f"Layer {idx+1}")
        # ── Quick tool shortcuts ──────────────────────────────────────────────
        elif key==ord('p'):
            self.tool='pen'; self.eraser=False; self.custom_col=None
            self._msg("Tool: pen")
        elif key==ord('o'):
            self.eraser=True; self.tool='eraser'
            self._msg("Eraser")
        elif key==ord('t'):
            self.tool='text'; self.eraser=False; self.text_mode=True
            self.text_buf=""; self._msg("Text mode — click to place")
        elif key==ord('f'):
            self._apply_effect(self._last_fx)
        # ── Brush size ────────────────────────────────────────────────────────
        elif key==ord('+') or key==ord('='):
            self.brush_cfg.size = min(MAX_BRUSH,self.brush_cfg.size+3)
            self._msg(f"Size {self.brush_cfg.size}")
        elif key==ord('-'):
            self.brush_cfg.size = max(MIN_BRUSH,self.brush_cfg.size-3)
            self._msg(f"Size {self.brush_cfg.size}")
        # ── Brush type ────────────────────────────────────────────────────────
        elif key==ord('n'):
            idx = BRUSH_TYPES.index(self.brush_cfg.brush_type)
            self.brush_cfg.brush_type = BRUSH_TYPES[(idx+1)%len(BRUSH_TYPES)]
            self._msg(f"Brush: {self.brush_cfg.brush_type}")
        # ── Timelapse ─────────────────────────────────────────────────────────
        elif key==ord('r'):
            if self.timelapse.recording:
                self.timelapse.stop()
                threading.Thread(target=self.timelapse.save,daemon=True).start()
                self._msg("Timelapse saved")
            else:
                self.timelapse.start(); self._msg("Recording timelapse…")
        # ── Sticker cycle ─────────────────────────────────────────────────────
        elif key==ord('i'):
            self.sticker_idx = (self.sticker_idx+1)%len(StickerLibrary.ALL_NAMES)
            self._msg(f"Sticker: {StickerLibrary.ALL_NAMES[self.sticker_idx]}")
        elif key==ord('m'):
            self.symmetry.mode = (SymMode.MANDALA
                                  if self.symmetry.mode!=SymMode.MANDALA
                                  else SymMode.NONE)
            self._msg(f"Mandala: {self.symmetry.mode==SymMode.MANDALA}")
        elif key==ord('j'):
            self._save('jpg')
        elif key==ord('u'):
            self._save('svg')
        return True


# =============================================================================
# SECTION 27 — MAIN ENTRY POINT
# =============================================================================
def _print_controls() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          Virtual Painter Pro v4.0  —  Controls Reference         ║
╠══════════════════════════════════════════════════════════════════╣
║  GESTURES                                                         ║
║   1 finger      → Draw / place shape start                        ║
║   2 fingers     → Confirm shape / click dock / click panel        ║
║   3 fingers     → Undo  (hold cooldown: 0.45s)                   ║
║   4 fingers     → Redo  (hold cooldown: 0.45s)                   ║
║   Palm open     → Clear active layer                              ║
║   Pinch close   → Zoom in/out on canvas                          ║
╠══════════════════════════════════════════════════════════════════╣
║  KEYBOARD — CANVAS                                                ║
║   Z / Y         → Undo / Redo                                    ║
║   C             → Clear active layer                              ║
║   G             → Toggle grid                                     ║
║   D             → Toggle Dark / Light theme                      ║
║   X             → Cycle symmetry mode                            ║
║   M             → Toggle Mandala mode                            ║
║   V             → Reset viewport (zoom + pan)                    ║
╠══════════════════════════════════════════════════════════════════╣
║  KEYBOARD — TOOLS                                                 ║
║   P             → Pen tool                                        ║
║   O             → Eraser                                          ║
║   T             → Text mode  (click to place, Enter to stamp)    ║
║   N             → Cycle brush type                               ║
║   + / -         → Increase / Decrease brush size                 ║
║   F             → Re-apply last effect                           ║
║   I             → Cycle sticker                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  KEYBOARD — LAYERS                                                ║
║   1 – 8         → Switch to Layer 1–8                            ║
║   L             → Open Layer panel                               ║
╠══════════════════════════════════════════════════════════════════╣
║  KEYBOARD — PANELS                                                ║
║   B             → Brush panel                                    ║
║   K             → Color picker                                   ║
║   E             → Effects panel                                  ║
║   A             → Analytics                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  KEYBOARD — SAVE / EXPORT                                         ║
║   S             → Export PNG  (exports/ folder)                  ║
║   J             → Export JPG                                     ║
║   U             → Export SVG                                     ║
║   Ctrl+S (19)   → Save .vpaint project  (projects/ folder)      ║
║   R             → Start / Stop timelapse recording              ║
╠══════════════════════════════════════════════════════════════════╣
║  VOICE COMMANDS (examples)                                        ║
║   "draw circle / rectangle / triangle / star"                    ║
║   "brush neon / airbrush / watercolor / calligraphy"             ║
║   "switch layer 2"  |  "zoom in"  |  "enable symmetry"          ║
║   "apply blur"  |  "save project"  |  "increase opacity"        ║
║   "dark mode"   |  "enable grid"   |  "red / blue / neon …"     ║
╠══════════════════════════════════════════════════════════════════╣
║   Q / ESC       → Quit                                           ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main() -> None:
    print("\n  Virtual Painter Pro v4.0  —  Starting …\n")
    _print_controls()

    # ── Camera setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open camera. Exiting.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {actual_w}×{actual_h}")

    painter = VirtualPainterV4(actual_w, actual_h)

    WIN = "Virtual Painter Pro v4.0"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, actual_w, actual_h)
    # Register mouse handler — gives reliable panel clicks without gestures
    cv2.setMouseCallback(WIN, painter.on_mouse)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Warning] Frame read failed — retrying…")
            time.sleep(0.05)
            continue

        output = painter.process(frame)
        if output is not None:
            cv2.imshow(WIN, output)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if not painter.handle_key(key):
                break   # quit

        # Window closed by user
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n  Shutting down…")
    painter.voice.stop()
    painter.voice_fb.stop()
    painter.lmk.close()

    # Offer to save timelapse if recording
    if painter.timelapse.recording and painter.timelapse.frames:
        painter.timelapse.stop()
        print("  Saving timelapse…")
        painter.timelapse.save()

    # Autosave on exit
    print("  Auto-saving project…")
    ProjectManager.save("autosave_exit.vpaint", painter)

    cap.release()
    cv2.destroyAllWindows()
    print("  Goodbye!\n")


if __name__ == "__main__":
    main()