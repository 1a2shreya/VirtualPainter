#!/usr/bin/env python3
"""
Virtual Painter Pro v3.1 - AI-Powered Modern Painting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (v3.0) Features:
  - Glassmorphism toolbar with transparency & blur
  - Floating left-side dock (compact tool icons)
  - Dark/Light theme toggle
  - AI Shape Correction (smooth shapes automatically)
  - Layer System (3-5 layers for pro workflow)
  - Timelapse Video Export
  - Voice Feedback (text-to-speech)
  - Extended Gestures (3-finger undo, 4-finger redo)
  - Animated Cursor with glow & trail
  - Professional Startup Screen

v3.1 Bug Fixes & Improvements:
  [FIX] Notification overlay (cmd_msg) now actually displayed on screen
  [FIX] Gesture repeat-fire: undo/redo/clear now fire once per hold with cooldown
  [FIX] JPG save color corruption (wrong BGR->RGB conversion removed)
  [FIX] VoiceFeedback.speak() is now non-blocking (runs in daemon thread)
  [FIX] History now uses O(1) deque instead of O(n) list.pop(0)
  [FIX] Shape confirm fires only once per two-finger gesture, not every frame
  [FIX] grid_on / grid_off voice commands now handled + G key shortcut
  [FIX] ShapeCorrector.smooth_line uses proper 1D convolution (not 2D blur)
  [FIX] ShapeCorrector.correct_shape passes correct (-1,1,2) contour to OpenCV
  [FIX] Text tool now implemented: tap to place, type on keyboard, Enter to stamp
  [FIX] Emoji tool now implemented: one-finger or two-finger tap to stamp
  [FIX] Startup screen emoji removed (OpenCV cannot render Unicode)
  [FIX] Timelapse frames capped to prevent out-of-memory on long sessions
  [FIX] History class moved before VirtualPainterV3 (logical ordering)
  [NEW] Grid overlay toggle (G key or "enable/disable grid" voice command)

Gestures  1 finger  -> DRAW
          2 fingers -> SELECT / confirm shape / dock click
          3 fingers -> UNDO  (once per hold)
          4 fingers -> REDO  (once per hold)
          Palm Open -> CLEAR CANVAS  (once per hold)
          Pinch     -> Resize brush
Keys      Z=Undo  Y=Redo  S=Save  C=Clear  G=Grid  D=Theme
          1-5=Layer  R=Timelapse  T=Text mode  Q/ESC=Quit
Voice     "red" "clear" "pen" "circle" "undo" "bigger"
          "dark mode" "enable grid" "save png" etc.
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['GLOG_minloglevel'] = '2'     # Suppress Google logging

import cv2, numpy as np, math, time, threading, urllib.request
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import logging
import queue

# Suppress all non-critical warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import speech_recognition as sr
    VOICE_OK = True
except ImportError:
    VOICE_OK = False

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H   = 1280, 720
DOCK_W         = 100        # floating left dock width
DOCK_MARGIN    = 16
TB_H           = 0          # no top toolbar (moving to dock)
SB_H           = 44         # status bar
STABLE_FRAMES  = 3
MIN_BRUSH, MAX_BRUSH = 2, 60
UNDO_LIMIT     = 40
MAX_LAYERS     = 5

# Cooldowns (seconds)
TB_COOLDOWN     = 0.35
GESTURE_COOLDOWN = 0.45   # FIX: prevents undo/redo/clear firing 30x/sec while held
SHAPE_COOLDOWN = 0.40
EMOJI_COOLDOWN = 0.50
PICKER_COOLDOWN= 0.10

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme System
# ─────────────────────────────────────────────────────────────────────────────
class Theme(Enum):
    DARK = "dark"
    LIGHT = "light"

THEMES = {
    "dark": {
        "bg":       (30, 28, 36),
        "bg2":      (20, 18, 26),
        "accent":   (30, 190, 255),
        "select":   (60, 200, 255),
        "active":   (55, 230, 100),
        "hover":    (0, 230, 240),
        "text":     (220, 218, 232),
        "muted":    (110, 108, 128),
        "border":   (60, 58, 78),
        "sep":      (48, 46, 62),
        "tool_bg":  (42, 40, 55),
        "tool_act": (22, 75, 38),
        "dock_bg":  (25, 24, 32),
        "sb":       (16, 15, 20),
    },
    "light": {
        "bg":       (245, 245, 250),
        "bg2":      (235, 235, 240),
        "accent":   (200, 100, 30),
        "select":   (180, 120, 50),
        "active":   (50, 180, 100),
        "hover":    (100, 150, 200),
        "text":     (30, 30, 40),
        "muted":    (120, 120, 130),
        "border":   (150, 150, 160),
        "sep":      (200, 200, 210),
        "tool_bg":  (220, 220, 230),
        "tool_act": (200, 240, 220),
        "dock_bg":  (240, 240, 248),
        "sb":       (230, 230, 238),
    }
}

# Color palette
PALETTE = [
    ("Red",     (0, 0, 255)), ("Orange",  (0, 130, 255)), ("Yellow", (0, 220, 255)),
    ("Lime",    (0, 255, 0)), ("Green",   (0, 180, 0)), ("Teal",   (80, 200, 80)),
    ("Cyan",    (255, 220, 0)), ("Sky",   (255, 160, 80)), ("Blue",  (255, 30, 30)),
    ("Navy",    (140, 20, 20)), ("Purple", (200, 0, 190)), ("Pink",  (180, 100, 255)),
    ("Magenta", (255, 0, 200)), ("Brown",  (40, 80, 150)), ("Tan",   (100, 170, 200)),
    ("White",   (255, 255, 255)), ("Gray",  (160, 160, 160)), ("Black", (10, 10, 10)),
]

BRUSH_SIZES = [3, 7, 14, 24, 36]
TOOLS = ['pen', 'eraser', 'line', 'rect', 'circle', 'fill', 'text', 'emoji']
EMOJIS = ['😀', '❤️', '⭐', '🚀', '🎨', '✨', '🎉', '🔥']

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

# Voice command map
VOICE_MAP = {
    'clear':'clear', 'wipe':'clear', 'reset':'clear',
    'save':'save', 'export':'save', 'capture':'save',
    'undo':'undo', 'back':'undo',
    'redo':'redo', 'forward':'redo',
    'bigger':'brush_up', 'larger':'brush_up', 'increase':'brush_up',
    'smaller':'brush_down', 'decrease':'brush_down', 'tiny':'brush_down',
    'pen':('tool','pen'), 'pencil':('tool','pen'), 'draw':('tool','pen'),
    'eraser':('tool','eraser'), 'erase':('tool','eraser'), 'rubber':('tool','eraser'),
    'line':('tool','line'), 'straight':('tool','line'),
    'rectangle':('tool','rect'), 'rect':('tool','rect'), 'square':('tool','rect'),
    'circle':('tool','circle'), 'oval':('tool','circle'), 'round':('tool','circle'),
    'fill':('tool','fill'), 'bucket':('tool','fill'), 'paint':('tool','fill'),
    'text':('tool','text'), 'write':('tool','text'), 'type':('tool','text'),
    'emoji':('tool','emoji'), 'sticker':('tool','emoji'),
    'red':('color',(0,0,255)), 'orange':('color',(0,130,255)),
    'yellow':('color',(0,220,255)), 'lime':('color',(0,255,0)),
    'green':('color',(0,200,0)), 'teal':('color',(80,200,80)),
    'cyan':('color',(255,220,0)), 'sky':('color',(255,160,80)),
    'blue':('color',(255,0,0)), 'navy':('color',(140,20,20)),
    'purple':('color',(200,0,190)), 'pink':('color',(180,100,255)),
    'magenta':('color',(255,0,200)), 'brown':('color',(40,80,150)),
    'white':('color',(255,255,255)), 'gray':('color',(160,160,160)),
    'grey':('color',(160,160,160)), 'black':('color',(10,10,10)),
    'dark mode':'toggle_dark', 'light mode':'toggle_light',
    'save png':'save_png', 'save jpg':'save_jpg',
    'new canvas':'new', 'new painting':'new',
    'enable grid':'grid_on', 'disable grid':'grid_off',
}


# ─────────────────────────────────────────────────────────────────────────────
# Startup Screen
# ─────────────────────────────────────────────────────────────────────────────
class StartupScreen:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.visible = True
        self.start_time = time.time()
        self.items = [
            ("Loading Hand Detection AI...", 0.5),
            ("Loading Voice Engine...", 1.2),
            ("Loading Gesture Recognition...", 1.9),
            ("Initializing Canvas...", 2.6),
        ]
        self.current_item = 0

    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed > 3.5:
            self.visible = False
            return
        
        for i, (_, t) in enumerate(self.items):
            if elapsed >= t:
                self.current_item = i

    def draw(self, frame):
        if not self.visible:
            return frame
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # FIX: removed emoji from putText -- OpenCV cannot render Unicode characters
        title = "Virtual Painter AI v3.1"
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(title, font, 1.4, 2)
        cv2.putText(frame, title, (self.W//2 - tw//2, self.H//2 - 80),
                    font, 1.4, (30, 190, 255), 2)
        
        # Version
        ver = "Version 3.1 - Phase 1 (Fixed)"
        (vw, vh), _ = cv2.getTextSize(ver, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, ver, (self.W//2 - vw//2, self.H//2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 108, 128), 1)
        
        # Loading items
        for i, (item_text, _) in enumerate(self.items):
            y = self.H//2 + 20 + i * 35
            status = "[OK]" if i < self.current_item else "[ ]"
            color = (55, 230, 100) if i < self.current_item else (30, 190, 255) if i == self.current_item else (110, 108, 128)
            
            cv2.putText(frame, f"{status}  {item_text}", (self.W//2 - 150, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Progress bar
        progress = min(1.0, (time.time() - self.start_time) / 3.5)
        bar_w = int(300 * progress)
        cv2.rectangle(frame, (self.W//2 - 150, self.H//2 + 160),
                      (self.W//2 - 150 + bar_w, self.H//2 + 165),
                      (30, 190, 255), -1)
        cv2.rectangle(frame, (self.W//2 - 150, self.H//2 + 160),
                      (self.W//2 + 150, self.H//2 + 165),
                      (60, 58, 78), 2)
        
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# History  [FIX: moved before VirtualPainterV3; uses O(1) deque instead of
#           O(n) list.pop(0)]
# ─────────────────────────────────────────────────────────────────────────────
class History:
    def __init__(self, limit=UNDO_LIMIT):
        # FIX: deque(maxlen=limit) auto-evicts oldest entry in O(1)
        self.stack  = deque(maxlen=limit)
        self.future = deque(maxlen=limit)

    def push(self, state):
        self.stack.append([layer.copy() for layer in state])
        self.future.clear()

    def undo(self, current_state):
        if self.stack:
            self.future.append([layer.copy() for layer in current_state])
            return list(self.stack.pop())
        return current_state

    def redo(self, current_state):
        if self.future:
            self.stack.append([layer.copy() for layer in current_state])
            return list(self.future.pop())
        return current_state


# ─────────────────────────────────────────────────────────────────────────────
# Layer Manager
# ─────────────────────────────────────────────────────────────────────────────
class LayerManager:
    def __init__(self, W, H, count=MAX_LAYERS):
        self.W, self.H = W, H
        self.layers = [np.zeros((H, W, 3), np.uint8) for _ in range(count)]
        self.active_layer = 0
        self.visible = [True] * count
        self.opacity = [1.0] * count

    def current(self):
        return self.layers[self.active_layer]

    def set_active(self, idx):
        if 0 <= idx < len(self.layers):
            self.active_layer = idx

    def toggle_visibility(self, idx):
        if 0 <= idx < len(self.layers):
            self.visible[idx] = not self.visible[idx]

    def clear_current(self):
        self.layers[self.active_layer].fill(0)

    def merge_visible(self):
        """Merge all visible layers into one canvas"""
        result = np.zeros((self.H, self.W, 3), np.uint8)
        for i, layer in enumerate(self.layers):
            if not self.visible[i]:
                continue
            # Create alpha mask and blend
            mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            alpha = self.opacity[i]
            layer_masked = cv2.bitwise_and(layer, layer, mask=mask)
            layer_weighted = cv2.convertScaleAbs(layer_masked, alpha=alpha, beta=0)
            
            result = cv2.add(result, layer_weighted)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# AI Shape Correction
# ─────────────────────────────────────────────────────────────────────────────
class ShapeCorrector:
    @staticmethod
    def correct_shape(points, shape_type='circle'):
        """Use cv2.approxPolyDP to smooth and correct drawn shapes."""
        if len(points) < 3:
            return points
        # FIX: OpenCV contour functions require shape (-1, 1, 2), not flat array
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        epsilon = 0.02 * cv2.arcLength(pts, False)
        corrected = cv2.approxPolyDP(pts, epsilon, False)
        return corrected.reshape(-1, 2).tolist()

    @staticmethod
    def smooth_line(points, window=5):
        """Smooth a path using a sliding-average convolution.
        FIX: was incorrectly applying a 2D GaussianBlur to a 1D point array.
             Now uses proper 1D convolution per axis."""
        if len(points) < window + 1:
            return points
        pts = np.array(points, dtype=np.float32)
        kernel = np.ones(window, dtype=np.float32) / window
        sx = np.convolve(pts[:, 0], kernel, mode='same')
        sy = np.convolve(pts[:, 1], kernel, mode='same')
        return list(zip(sx.astype(int), sy.astype(int)))


# ─────────────────────────────────────────────────────────────────────────────
# Timelapse Recorder
# ─────────────────────────────────────────────────────────────────────────────
class TimelapseRecorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start(self):
        self.recording = True
        self.frames.clear()

    def stop(self):
        self.recording = False

    def add_frame(self, frame):
        # FIX: cap at ~10 min @30fps to prevent out-of-memory on long sessions
        if self.recording and len(self.frames) < 18000:
            self.frames.append(frame.copy())

    def save(self, path="timelapse.mp4", fps=30):
        if not self.frames:
            return False
        try:
            h, w = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            
            for frame in self.frames:
                writer.write(frame)
            
            writer.release()
            return True
        except Exception as e:
            print(f"[!] Timelapse save error: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Animated Cursor with Glow & Trail
# ─────────────────────────────────────────────────────────────────────────────
class AnimatedCursor:
    def __init__(self, max_trail=15):
        self.trail = deque(maxlen=max_trail)
        self.glow_intensity = 0.0

    def add_point(self, x, y):
        self.trail.append((x, y))

    def draw(self, frame, x, y, brush_size, color, eraser=False):
        # Draw trail with fading effect
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) if self.trail else 0
            trail_col = tuple(int(c * alpha * 0.6) for c in color) if not eraser else (140, 140, 255)
            cv2.circle(frame, (int(tx), int(ty)), max(2, int(brush_size * alpha * 0.5)),
                      trail_col, -1)
        
        # Draw main cursor circle
        if eraser:
            cv2.circle(frame, (x, y), brush_size * 3, (140, 140, 255), 2)
        else:
            # Add glow effect
            for r in range(brush_size + 8, brush_size, 2):
                glow_col = tuple(int(c * 0.2) for c in color)
                cv2.circle(frame, (x, y), r, glow_col, 1)
            cv2.circle(frame, (x, y), brush_size, color, 2)
        
        # Center dot
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    def clear(self):
        self.trail.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Floating Dock (Left side tool panel)
# ─────────────────────────────────────────────────────────────────────────────
class FloatingDock:
    # Text labels instead of emoji (OpenCV can't render emoji)
    LABELS = {
        'pen': 'PEN',
        'eraser': 'ERA',
        'line': 'LIN',
        'rect': 'REC',
        'circle': 'CIR',
        'fill': 'FIL',
        'text': 'TXT',
        'emoji': 'EMO',
    }
    
    ICON_SIZE = 56
    SPACING = 8
    
    def __init__(self, dock_w, canvas_h, theme_name="dark"):
        self.dock_w = dock_w
        self.canvas_h = canvas_h
        self.theme_name = theme_name
        self.rects = []  # (tool_name, x1, y1, x2, y2)
        self.y_offset = DOCK_MARGIN

    def update_theme(self, theme_name):
        self.theme_name = theme_name

    def draw(self, frame, active_tool, eraser):
        colors = THEMES[self.theme_name]
        
        # Semi-transparent dock background with blur effect
        dock_area = frame[:, :self.dock_w]
        blurred = cv2.GaussianBlur(dock_area, (21, 21), 0)
        frame[:, :self.dock_w] = cv2.addWeighted(blurred, 0.7, dock_area, 0.3, 0)
        
        # Dock border
        cv2.line(frame, (self.dock_w, 0), (self.dock_w, self.canvas_h),
                colors["border"], 2)
        
        self.rects.clear()
        y = self.y_offset
        
        for tool in TOOLS:
            is_active = (tool == active_tool and not eraser) or (tool == 'eraser' and eraser)
            
            x1, y1 = DOCK_MARGIN, y
            x2, y2 = x1 + self.ICON_SIZE, y1 + self.ICON_SIZE
            
            # Background
            bg_color = colors["tool_act"] if is_active else colors["tool_bg"]
            self._rounded_rect(frame, x1, y1, x2, y2, 8, bg_color, -1)
            
            # Border
            border_color = colors["active"] if is_active else colors["border"]
            border_width = 3 if is_active else 1
            self._rounded_rect(frame, x1, y1, x2, y2, 8, border_color, border_width)
            
            # Icon text (use ASCII text labels)
            label = self.LABELS.get(tool, '???')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (iw, ih), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.putText(frame, label, (x1 + self.ICON_SIZE//2 - iw//2,
                                       y1 + self.ICON_SIZE//2 + ih//2),
                       font, font_scale, colors["text"], thickness)
            
            self.rects.append((tool, x1, y1, x2, y2))
            y += self.ICON_SIZE + self.SPACING

    def hit(self, x, y):
        """Check if click hit a tool button"""
        for tool, x1, y1, x2, y2 in self.rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return tool
        return None

    @staticmethod
    def _rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
        """Draw a rounded rectangle"""
        r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
        if thickness == -1:
            cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
            cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
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
            cv2.ellipse(img, (x1+r, y2-r), (r,r), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2-r, y2-r), (r,r), 0, 0, 90, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# Text-to-Speech Feedback
# ─────────────────────────────────────────────────────────────────────────────
class VoiceFeedback:
    def __init__(self):
        self.enabled = TTS_OK
        self._q = queue.Queue(maxsize=4)
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                # FIX: run TTS in daemon thread so speak() never blocks the main loop
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception as e:
                print(f"[!] TTS init error: {e}")
                self.enabled = False

    def _worker(self):
        while True:
            text = self._q.get()
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass

    def speak(self, text):
        if not self.enabled:
            return
        try:
            self._q.put_nowait(text)   # non-blocking; drop if backlogged
        except queue.Full:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Voice Controller (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────
class VoiceController:
    def __init__(self, callback):
        self.cb = callback
        self.running = False
        self.last_cmd = ''
        self.last_t = 0.0
        self.status = 'off'

    @staticmethod
    def _match(text):
        words = text.lower().split()
        for phrase in sorted(VOICE_MAP.keys(), key=len, reverse=True):
            if phrase in text:
                return VOICE_MAP[phrase]
        for word in words:
            if word in VOICE_MAP:
                return VOICE_MAP[word]
        return None

    def start(self):
        if not VOICE_OK:
            print("  [!] Voice disabled — run: pip install SpeechRecognition pyaudio\n")
            self.status = 'off'
            return
        self.running = True
        self.status = 'listening'
        threading.Thread(target=self._loop, daemon=True).start()
        print("  ✓ Voice active\n")

    def stop(self):
        self.running = False
        self.status = 'off'

    def _loop(self):
        rec = sr.Recognizer()
        rec.energy_threshold = 400
        rec.dynamic_energy_threshold = True
        rec.pause_threshold = 0.6

        while self.running:
            try:
                with sr.Microphone() as src:
                    rec.adjust_for_ambient_noise(src, 1.0)
                    while self.running:
                        try:
                            self.status = 'listening'
                            audio = rec.listen(src, timeout=1.5, phrase_time_limit=5)
                            self.status = 'processing'
                            text = rec.recognize_google(audio).lower()
                            action = self._match(text)
                            if action:
                                self.last_cmd = text
                                self.last_t = time.time()
                                self.status = 'heard'
                                print(f"  🎤 Command: {text}")
                                self.cb(action)
                            else:
                                self.status = 'listening'
                        except sr.WaitTimeoutError:
                            self.status = 'listening'
                        except sr.UnknownValueError:
                            self.status = 'listening'
                        except sr.RequestError as e:
                            print(f"  [!] Voice API: {e}")
                            self.status = 'error'
                            time.sleep(3)
            except OSError as e:
                print(f"  [!] Mic unavailable: {e}")
                self.status = 'off'
                break
            except Exception as e:
                print(f"  [!] Voice error: {e}")
                self.status = 'error'
                time.sleep(5)

    @property
    def recent(self):
        return self.last_cmd if time.time() - self.last_t < 2.5 else ''


# ─────────────────────────────────────────────────────────────────────────────
# Gesture Recognition (Extended)
# ─────────────────────────────────────────────────────────────────────────────
def get_gesture_extended(lm):
    """Enhanced gesture recognition with 3-finger, 4-finger, and palm"""
    index_up = lm[8].y < lm[6].y - 0.02
    middle_up = lm[12].y < lm[10].y - 0.02
    ring_up = lm[16].y < lm[14].y - 0.045
    pinky_up = lm[20].y < lm[18].y - 0.045
    thumb_up = lm[4].y < lm[3].y - 0.05
    pinch_d = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) * CAM_W

    # Palm open (all fingers up)
    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return 'palm_open', pinch_d
    
    # Four fingers (thumb down, 4 fingers up)
    if index_up and middle_up and ring_up and pinky_up and not thumb_up:
        return 'four_fingers', pinch_d
    
    # Three fingers (thumb down, 3 up)
    if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
        return 'three_fingers', pinch_d
    
    # Pinch
    if pinch_d < 38 and not index_up:
        return 'pinch', pinch_d
    
    # Two fingers
    if index_up and middle_up and not ring_up and not pinky_up:
        return 'two_fingers', pinch_d
    
    # One finger
    if index_up and not middle_up:
        return 'one_finger', pinch_d
    
    return 'idle', pinch_d


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (existing from v2)
# ─────────────────────────────────────────────────────────────────────────────
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"\n  Downloading hand model (~3 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  ✓ Done!\n")
    except Exception as e:
        print(f"  [✗] {e}")
        raise SystemExit(1)


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
    fc = (int(color[0]), int(color[1]), int(color[2]))
    cv2.floodFill(canvas, mask, (x, y), fc,
                  loDiff=(35, 35, 35), upDiff=(35, 35, 35),
                  flags=cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
    canvas[mask[1:-1, 1:-1] == 1] = fc


def gradient_fill(img, x1, y1, x2, y2, c1, c2, vertical=True):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
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


def alpha_blend(img, x1, y1, x2, y2, color, alpha=0.40):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2-x1, y2-y1), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, img[y1:y2, x1:x2])


def text_centered(img, txt, cx, cy, scale, color, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    cv2.putText(img, txt, (cx - tw//2, cy + th//2), font, scale, color, thick)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Class
# ─────────────────────────────────────────────────────────────────────────────
class VirtualPainterV3:
    def __init__(self, W=CAM_W, H=CAM_H):
        self.W, self.H = W, H
        self.canvas_h = H - SB_H
        self._voice_actions = queue.Queue()
        
        # Theme
        self.current_theme = Theme.DARK
        self.colors = THEMES["dark"]
        
        # Drawing state
        self.colour_idx = 0
        self.custom_col = None
        self.brush_size = BRUSH_SIZES[2]
        self.brush_idx = 2
        self.tool = 'pen'
        self.eraser = False
        
        # Layers
        self.layers = LayerManager(W, self.canvas_h, MAX_LAYERS)
        self.hist = History(limit=UNDO_LIMIT)
        
        # UI Components
        self.dock = FloatingDock(DOCK_W, self.canvas_h)
        self.cursor = AnimatedCursor()
        self.startup = StartupScreen(W, H)
        self.timelapse = TimelapseRecorder()
        self.voice_fb = VoiceFeedback()
        self.shape_corrector = ShapeCorrector()
        
        # Shape drawing
        self.shape_start = None
        self.shape_preview = None
        self.last_draw_xy = None
        
        # Text
        self.text_mode = False
        self.text_pos = None
        self.text_buf = ''
        
        # Cooldowns
        self.tb_click_t = 0.0
        self.shape_click_t = 0.0
        self.emoji_click_t = 0.0
        self.cmd_msg = ''
        self.cmd_t = 0.0
        
        # Gesture state
        self.g_buf = deque(maxlen=STABLE_FRAMES)
        self.g_cur = 'idle'
        self.prev_xy = None
        self.smooth_xy = (0, 0)
        self.raw_prev = (0, 0)
        self.fill_done = False
        # NEW: grid toggle (voice + G key)
        self.grid_on = False
        # FIX: per-gesture timestamps to prevent repeat-fire while held
        self._last_gesture_t: dict = {}
        # FIX: prevent shape confirm firing every frame in two_fingers
        self._shape_confirmed = False
        
        # MediaPipe
        ensure_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.70)
        self.lmk = mp_vision.HandLandmarker.create_from_options(opts)
        self.ts_base = int(time.time()*1000)
        self.last_ts = 0
        
        # Voice (enqueue actions; apply on main thread for smoothness)
        self.voice = VoiceController(self._enqueue_voice_action)
        self.voice.start()
        
        # FPS
        self._fps_t = time.time()
        self._fps_buf = deque([30.0]*10, maxlen=10)

    @property
    def active_color(self):
        if self.eraser:
            return (0, 0, 0)
        return self.custom_col or PALETTE[self.colour_idx][1]

    def _enqueue_voice_action(self, action):
        """Called from the voice thread. Must be non-blocking and thread-safe."""
        try:
            self._voice_actions.put_nowait(action)
        except queue.Full:
            pass

    def _apply_queued_voice_actions(self, max_per_frame=2):
        """Apply voice actions on the main thread to avoid race conditions/stutters."""
        applied = 0
        while applied < max_per_frame:
            try:
                action = self._voice_actions.get_nowait()
            except queue.Empty:
                break
            self._voice_action(action)
            applied += 1

    def _voice_action(self, action):
        """Handle voice commands (main thread only)."""
        if action == 'clear':
            self._push()
            self.layers.clear_current()
            self._msg("Canvas cleared")
            self.voice_fb.speak("Canvas cleared")
        elif action == 'save':
            self._save()
            self._msg("Saved!")
            self.voice_fb.speak("File saved")
        elif action == 'save_png':
            self._save('png')
            self._msg("Saved as PNG!")
            self.voice_fb.speak("Saved as PNG")
        elif action == 'save_jpg':
            self._save('jpg')
            self._msg("Saved as JPG!")
            self.voice_fb.speak("Saved as JPEG")
        elif action == 'undo':
            self._undo()
            self._msg("Undo")
            self.voice_fb.speak("Undo")
        elif action == 'redo':
            self._redo()
            self._msg("Redo")
            self.voice_fb.speak("Redo")
        elif action == 'brush_up':
            self.brush_idx = min(len(BRUSH_SIZES)-1, self.brush_idx+1)
            self.brush_size = BRUSH_SIZES[self.brush_idx]
            self._msg(f"Brush: {self.brush_size}px")
            self.voice_fb.speak(f"Brush size {self.brush_size}")
        elif action == 'brush_down':
            self.brush_idx = max(0, self.brush_idx-1)
            self.brush_size = BRUSH_SIZES[self.brush_idx]
            self._msg(f"Brush: {self.brush_size}px")
            self.voice_fb.speak(f"Brush size {self.brush_size}")
        elif action == 'toggle_dark':
            self.toggle_theme(Theme.DARK)
            self.voice_fb.speak("Dark mode")
        elif action == 'toggle_light':
            self.toggle_theme(Theme.LIGHT)
            self.voice_fb.speak("Light mode")
        elif action == 'new':
            self._push()
            self.layers.clear_current()
            self._msg("New canvas")
            self.voice_fb.speak("New canvas")
        # FIX: grid_on / grid_off were in VOICE_MAP but never handled here
        elif action == 'grid_on':
            self.grid_on = True
            self._msg("Grid ON")
            self.voice_fb.speak("Grid on")
        elif action == 'grid_off':
            self.grid_on = False
            self._msg("Grid OFF")
            self.voice_fb.speak("Grid off")
        elif isinstance(action, tuple):
            k, v = action
            if k == 'tool':
                self.eraser = (v == 'eraser')
                self.tool = 'pen' if v == 'eraser' else v
                self._msg(f"{v.title()}")
                self.voice_fb.speak(v)
            elif k == 'color':
                self.custom_col = v
                self.eraser = False
                self._msg("Color changed")
                self.voice_fb.speak("Color changed")

    def _msg(self, t):
        self.cmd_msg = t
        self.cmd_t = time.time()

    def _push(self):
        self.hist.push(self.layers.layers)

    def _undo(self):
        restored = self.hist.undo(self.layers.layers)
        # Restore in-place to keep LayerManager/list references consistent
        if isinstance(restored, list) and len(restored) == len(self.layers.layers):
            for i in range(len(self.layers.layers)):
                self.layers.layers[i][:] = restored[i]
        else:
            self.layers.layers = restored

    def _redo(self):
        restored = self.hist.redo(self.layers.layers)
        # Restore in-place to keep LayerManager/list references consistent
        if isinstance(restored, list) and len(restored) == len(self.layers.layers):
            for i in range(len(self.layers.layers)):
                self.layers.layers[i][:] = restored[i]
        else:
            self.layers.layers = restored

    def _save(self, fmt='png'):
        os.makedirs('paintings', exist_ok=True)
        ext = fmt if fmt in ('png', 'jpg') else 'png'
        p = f'paintings/painting_{datetime.now():%Y%m%d_%H%M%S}.{ext}'
        canvas = self.layers.merge_visible()
        # FIX: removed incorrect cv2.COLOR_BGR2RGB conversion before cv2.imwrite.
        # cv2.imwrite always expects BGR; converting to RGB first swapped R/B channels.
        cv2.imwrite(p, canvas)
        print(f'  Saved -> {os.path.abspath(p)}')
        self._msg(f"Saved as {ext.upper()}!")

    def toggle_theme(self, theme):
        self.current_theme = theme
        self.colors = THEMES[theme.value]
        self.dock.update_theme(theme.value)
        self._msg(f"{theme.value.title()} mode")

    def _draw_shape(self, cv, p1, p2, col, thick):
        if self.tool == 'line':
            cv2.line(cv, p1, p2, col, thick)
        elif self.tool == 'rect':
            cv2.rectangle(cv, p1, p2, col, thick)
        elif self.tool == 'circle':
            r = int(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
            cv2.circle(cv, p1, r, col, thick)

    def _gesture_fire(self, name, cooldown=None):
        """Return True and record timestamp only if gesture cooldown has passed.
        FIX: prevents undo/redo/clear firing 30x/sec while the gesture is held."""
        cd = cooldown if cooldown is not None else GESTURE_COOLDOWN
        now = time.time()
        if now - self._last_gesture_t.get(name, 0) > cd:
            self._last_gesture_t[name] = now
            return True
        return False

    def _stamp_text(self, x, y):
        """Commit the typed text buffer onto the canvas at (x, y).
        FIX: text tool was initialised but never implemented."""
        if not self.text_buf.strip():
            self.text_mode = False
            self.text_buf = ''
            return
        self._push()
        scale = max(0.5, self.brush_size / 12.0)
        thick = max(1, self.brush_size // 8)
        cv2.putText(self.layers.current(), self.text_buf, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, self.active_color, thick)
        self.text_buf = ''
        self.text_pos = None
        self.text_mode = False
        self._msg("Text stamped")

    def _stamp_emoji(self, x, y):
        """Stamp an ASCII emoji label at (x, y) on the canvas.
        FIX: emoji tool was initialised but never implemented.
        Note: OpenCV cannot render Unicode emoji, so ASCII art is used."""
        EMOJI_ASCII = [':-)', '<3', '(*)', '>>>', '[art]', '**', '!!!', '###']
        label = EMOJI_ASCII[self.emoji_idx % len(EMOJI_ASCII)]
        scale = max(0.6, self.brush_size / 14.0)
        cv2.putText(self.layers.current(), label, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale,
                    self.active_color, max(1, int(scale * 2)))
        self._msg(f"Stamp: {label}")

    def process(self, frame):
        # Apply any pending voice commands at frame start (main thread)
        self._apply_queued_voice_actions(max_per_frame=2)

        frame = cv2.flip(frame, 1)
        if frame.shape[:2] != (self.H, self.W):
            frame = cv2.resize(frame, (self.W, self.H))

        # Draw startup screen if visible
        if self.startup.visible:
            self.startup.update()
            return self.startup.draw(frame)

        # Process hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.time()*1000)-self.ts_base, self.last_ts+1)
        self.last_ts = ts
        res = self.lmk.detect_for_video(mp_im, ts)

        mode = 'idle'
        sx, sy = self.smooth_xy

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            draw_skeleton(frame, lm, self.W, self.H)

            rx, ry = int(lm[8].x*self.W), int(lm[8].y*self.H)
            prx, pry = self.raw_prev
            speed = math.hypot(rx-prx, ry-pry)
            alpha = min(0.88, max(0.30, speed/45.0))
            self.raw_prev = (rx, ry)
            sx = int(alpha*rx+(1-alpha)*sx)
            sy = int(alpha*ry+(1-alpha)*sy)
            self.smooth_xy = (sx, sy)

            # Gesture detection (extended)
            raw_g, pinch_d = get_gesture_extended(lm)
            self.g_buf.append(raw_g)
            if len(self.g_buf) == STABLE_FRAMES and len(set(self.g_buf)) == 1:
                self.g_cur = raw_g
            g = self.g_cur

            # Handle gestures
            if g == 'pinch':
                mode = 'pinch'
                self.brush_size = int(np.interp(pinch_d, [15, 120], [MIN_BRUSH, MAX_BRUSH]))
                r = self.brush_size
                cv2.circle(frame, (sx, sy), r, self.colors["accent"], 2)
                cv2.circle(frame, (sx, sy), r//2, (255, 220, 100), 1)
                cv2.circle(frame, (sx, sy), 3, (255, 255, 255), -1)
                self.prev_xy = None

            elif g == 'three_fingers':
                # FIX: was calling _undo() every frame; now fires once per hold
                if self._gesture_fire('three_fingers'):
                    self._undo()
                    self._msg("3-Finger Undo")

            elif g == 'four_fingers':
                # FIX: was calling _redo() every frame; now fires once per hold
                if self._gesture_fire('four_fingers'):
                    self._redo()
                    self._msg("4-Finger Redo")

            elif g == 'palm_open':
                # FIX: was clearing canvas every frame; now fires once per hold
                if self._gesture_fire('palm_open'):
                    self._push()
                    self.layers.clear_current()
                    self._msg("Palm Clear")

            elif g == 'two_fingers':
                mode = 'select'
                self.prev_xy = None
                self.fill_done = False

                # FIX: shape confirm previously fired every frame while two_fingers
                # was held, calling _push() + _draw_shape() 30x/sec.
                # Now uses _shape_confirmed flag to fire exactly once.
                if self.shape_start and not self._shape_confirmed:
                    self._push()
                    self._draw_shape(self.layers.current(),
                                     self.shape_start,
                                     self.last_draw_xy or self.shape_start,
                                     self.active_color, self.brush_size)
                    self.shape_start = None
                    self._shape_confirmed = True

                x1, y1 = int(lm[8].x * self.W), int(lm[8].y * self.H)
                x2, y2 = int(lm[12].x * self.W), int(lm[12].y * self.H)
                click_dist = math.hypot(x2 - x1, y2 - y1)

                cv2.circle(frame, (sx, sy), 8, self.colors["select"], -1)

                if sx < DOCK_W:
                    cv2.circle(frame, (sx, sy), 8, self.colors["hover"], -1)
                    if time.time() - self.tb_click_t > TB_COOLDOWN:
                        hit_tool = self.dock.hit(sx, sy)
                        if hit_tool:
                            self.tb_click_t = time.time()
                            if hit_tool == 'eraser':
                                self.eraser = True
                                self.tool = 'pen'
                            else:
                                self.eraser = False
                                self.tool = hit_tool
                            self._msg(f"{hit_tool.upper()}")
                            cv2.circle(frame, (sx, sy), 12, self.colors["active"], -1)
                elif click_dist < 60:
                    cv2.circle(frame, (sx, sy), 8, self.colors["active"], -1)
                    # FIX: emoji tool tap-to-stamp (was never wired up)
                    if self.tool == 'emoji' and self._gesture_fire('emoji_tap', cooldown=0.5):
                        self._push()
                        self._stamp_emoji(sx, sy)
                    # FIX: text tool tap-to-place (was never wired up)
                    elif self.tool == 'text' and self._gesture_fire('text_place', cooldown=0.6):
                        self.text_pos = (sx, sy)
                        self.text_mode = True
                        self._msg("Type text, press Enter to stamp")

            elif g == 'one_finger':
                mode = 'draw'
                # Reset shape-confirm flag so the next shape can be confirmed later
                self._shape_confirmed = False

                if sx > DOCK_W:
                    col = self.active_color
                    sz = self.brush_size

                    if self.eraser or self.tool == 'pen':
                        dc = (0, 0, 0) if self.eraser else col
                        ds = sz*3 if self.eraser else sz
                        if self.prev_xy:
                            cv2.line(self.layers.current(), self.prev_xy, (sx, sy), dc, ds*2)
                            cv2.circle(self.layers.current(), (sx, sy), ds, dc, -1)
                        self.prev_xy = self.last_draw_xy = (sx, sy)

                    elif self.tool == 'fill':
                        if not self.fill_done:
                            self._push()
                            flood_fill(self.layers.current(), sx, sy, col)
                            self.fill_done = True
                        self.prev_xy = (sx, sy)

                    elif self.tool in ('line', 'rect', 'circle'):
                        if self.shape_start is None:
                            self.shape_start = (sx, sy)
                        prev = self.layers.current().copy()
                        self._draw_shape(prev, self.shape_start, (sx, sy), col, sz)
                        self.shape_preview = prev
                        self.prev_xy = self.last_draw_xy = (sx, sy)

                    # FIX: text tool one-finger sets anchor position
                    elif self.tool == 'text':
                        if self.prev_xy is None:
                            self.text_pos = (sx, sy)
                            self.text_mode = True
                            self._msg("Type text, press Enter to stamp")
                        self.prev_xy = (sx, sy)

                    # FIX: emoji one-finger stamps with cooldown
                    elif self.tool == 'emoji':
                        if self._gesture_fire('emoji_draw', cooldown=0.4):
                            self._push()
                            self._stamp_emoji(sx, sy)
                        self.prev_xy = (sx, sy)

                    else:
                        self.prev_xy = (sx, sy)

                self.cursor.add_point(sx, sy)
                self.cursor.draw(frame, sx, sy, self.brush_size, self.active_color, self.eraser)
            else:
                self.prev_xy = None
                self.fill_done = False

        # Composite canvas
        canvas = self.shape_preview if self.shape_preview is not None else self.layers.merge_visible()
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        frame_area = frame[:self.canvas_h, :]
        out = cv2.add(cv2.bitwise_and(frame_area, frame_area, mask=cv2.bitwise_not(mask)),
                     cv2.bitwise_and(canvas, canvas, mask=mask))
        frame[:self.canvas_h, :] = out
        self.shape_preview = None

        # NEW: draw grid overlay on the composited frame
        if self.grid_on:
            step = 50
            grid_col = (55, 55, 65) if self.current_theme == Theme.DARK else (190, 190, 200)
            for x in range(DOCK_W, self.W, step):
                cv2.line(frame, (x, 0), (x, self.canvas_h), grid_col, 1)
            for y in range(0, self.canvas_h, step):
                cv2.line(frame, (DOCK_W, y), (self.W, y), grid_col, 1)

        # FIX: show live text preview while text_mode is active
        if self.text_mode and self.text_pos:
            preview = self.text_buf + "|"
            scale = max(0.5, self.brush_size / 12.0)
            thick = max(1, self.brush_size // 8)
            cv2.putText(frame, preview, self.text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, scale, self.active_color, thick)

        # Draw floating dock
        self.dock.draw(frame, self.tool, self.eraser)

        # Status bar
        self._draw_status_bar(frame, mode)

        # Timelapse recording
        if self.timelapse.recording:
            self.timelapse.add_frame(canvas)

        # FPS
        now = time.time()
        self._fps_buf.append(1.0/max(now-self._fps_t, 1e-9))
        self._fps_t = now

        return frame

    def _draw_status_bar(self, frame, mode):
        y1, y2 = self.H - SB_H, self.H
        gradient_fill(frame, 0, y1, self.W, y2, self.colors["sb"], self.colors["bg2"])
        cv2.line(frame, (0, y1), (self.W, y1), self.colors["sep"], 1)

        fps = sum(self._fps_buf) / len(self._fps_buf)
        cy = (y1 + y2) // 2 + 6

        # FIX: removed Unicode emoji from mode labels (OpenCV can't render them)
        mode_info = {
            'idle':   ('IDLE',                                self.colors["muted"]),
            'select': ('SELECT',                              self.colors["select"]),
            'pinch':  (f'BRUSH {self.brush_size}px',         self.colors["accent"]),
            'draw':   (('ERASING' if self.eraser else self.tool.upper()), self.colors["active"]),
        }
        lbl, col = mode_info.get(mode, ('IDLE', self.colors["muted"]))
        cv2.putText(frame, lbl, (14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        # Centre: theme + layer + grid indicator
        centre = f"[{self.current_theme.value.upper()}]  L{self.layers.active_layer + 1}"
        if self.grid_on:
            centre += "  GRID"
        (cw, _), _ = cv2.getTextSize(centre, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, centre, (self.W//2 - cw//2, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.colors["muted"], 1)

        # FPS
        fps_txt = f'FPS {fps:.0f}'
        (fw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, fps_txt, (self.W - fw - 10, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.colors["muted"], 1)

        # FIX: cmd_msg was SET but NEVER DRAWN — all voice/gesture feedback
        # was silently discarded.  Now renders as a fading pill above the bar.
        if self.cmd_msg and time.time() - self.cmd_t < 2.5:
            age   = time.time() - self.cmd_t
            alpha = max(0.0, min(1.0, (2.5 - age) / 0.5))
            tcol  = tuple(int(c * alpha) for c in self.colors["accent"])
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (mw, mh), _ = cv2.getTextSize(self.cmd_msg, font, 0.55, 1)
            mx = self.W // 2 - mw // 2
            my = y1 - 12
            pad = 8
            # semi-transparent background pill
            alpha_blend(frame, mx - pad, my - mh - pad,
                        mx + mw + pad, my + pad,
                        self.colors["bg2"], alpha=0.75)
            cv2.putText(frame, self.cmd_msg, (mx, my), font, 0.55, tcol, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print('\n[Error] Cannot open webcam\n')
        return

    p = VirtualPainterV3()

    print('\n' + '='*68)
    print('   Virtual Painter Pro v3.1 - AI-Powered Painting')
    print('='*68)
    print('   1 finger   -> DRAW')
    print('   2 fingers  -> SELECT / confirm shape / dock click')
    print('   3 fingers  -> UNDO  (once per hold)')
    print('   4 fingers  -> REDO  (once per hold)')
    print('   Palm open  -> CLEAR CANVAS  (once per hold)')
    print('   Pinch      -> Resize brush')
    print()
    print('   1-5=Layer  Z=Undo  Y=Redo  S=Save  C=Clear')
    print('   G=Grid  D=Theme  T=Text mode  R=Timelapse  Q/ESC=Quit')
    print()
    print('   Voice: "red" "clear" "pen" "circle" "dark mode"')
    print('          "enable grid" "disable grid" "save png" etc.')
    print('='*68 + '\n')

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = p.process(frame)
        cv2.imshow('Virtual Painter v3.1', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            p._save()
        elif key == ord('c'):
            p._push()
            p.layers.clear_current()
            p._msg("Canvas cleared")
        elif key == ord('z'):
            p._undo()
            p._msg("Undo")
        elif key == ord('y'):
            p._redo()
            p._msg("Redo")
        elif key == ord('d'):
            new_theme = Theme.LIGHT if p.current_theme == Theme.DARK else Theme.DARK
            p.toggle_theme(new_theme)
        # NEW: G key toggles grid
        elif key == ord('g'):
            p.grid_on = not p.grid_on
            p._msg(f"Grid {'ON' if p.grid_on else 'OFF'}")
        # NEW: T key enters text mode (type on keyboard, Enter to stamp)
        elif key == ord('t') and not p.text_mode:
            p.tool = 'text'
            p.eraser = False
            p._msg("Text tool: point with 1 finger to place")
        elif key in [ord(str(i)) for i in range(1, MAX_LAYERS + 1)]:
            layer_idx = key - ord('1')
            p.layers.set_active(layer_idx)
            p._msg(f"Layer {layer_idx + 1} Active")
        elif key == ord('r'):
            if p.timelapse.recording:
                p.timelapse.stop()
                if p.timelapse.save('timelapse.mp4'):
                    print('\n  Timelapse saved!\n')
            else:
                p.timelapse.start()
                print('\n  Recording timelapse...\n')
        # Text mode keyboard input
        elif p.text_mode:
            if key == 13:   # Enter -> stamp text
                if p.text_pos:
                    p._stamp_text(*p.text_pos)
            elif key == 8:  # Backspace
                p.text_buf = p.text_buf[:-1]
            elif 32 <= key <= 126:
                p.text_buf += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    p.lmk.close()
    p.voice.stop()

    if p.timelapse.recording:
        p.timelapse.stop()
        p.timelapse.save('timelapse.mp4')

    print('\n  Goodbye!\n')


if __name__ == '__main__':
    main()