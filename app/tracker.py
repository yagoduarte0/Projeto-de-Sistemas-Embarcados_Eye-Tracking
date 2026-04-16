"""
Rastreamento de olhar usando MediaPipe FaceLandmarker (Tasks API).
Sem calibração. Sem modelo ML. Funciona direto da câmera.

Lógica de detecção:
  - Íris ratio  < 0.33 ou > 0.67 → olhando para o lado
  - Head yaw normalizado alto    → cabeça virada
  - EAR < threshold por > 2.5s  → perda de foco / sonolência
"""
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Landmarks MediaPipe Face Mesh (478 pontos com íris) ───────────────────────
R_IRIS, L_IRIS               = 468, 473   # centro de cada íris
R_EYE_LEFT,  R_EYE_RIGHT     =  33, 133   # cantos olho direito
L_EYE_LEFT,  L_EYE_RIGHT     = 362, 263   # cantos olho esquerdo
R_EYE_TOP,   R_EYE_BOTTOM    = 159, 145   # pálpebra direita
L_EYE_TOP,   L_EYE_BOTTOM    = 386, 374   # pálpebra esquerda
NOSE_TIP                     =   4        # ponta do nariz

# ── Thresholds ────────────────────────────────────────────────────────────────
IRIS_SIDE_LOW   = 0.33   # ratio < → olhando para esquerda
IRIS_SIDE_HIGH  = 0.67   # ratio > → olhando para direita
IRIS_V_LOW      = 0.25   # ratio vertical < → olhando para cima
IRIS_V_HIGH     = 0.75   # ratio vertical > → olhando para baixo
HEAD_YAW_THRESH = 0.18   # desvio normalizado do nariz
EAR_THRESHOLD   = 0.18   # EAR abaixo = olho fechado
BLINK_SUSTAINED_SECS   = 2.5
DISTRACTION_COOLDOWN   = 3.0
ALERT_DISTRACTION_SECS = 5.0
SMOOTH_FRAMES          = 5    # frames consecutivos para confirmar estado


# ── Dados de sessão ───────────────────────────────────────────────────────────

@dataclass
class Event:
    kind: str
    timestamp: float
    detail: str = ""


@dataclass
class SessionStats:
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_distractions: int = 0
    gaze_away_count: int = 0
    focus_lost_count: int = 0
    total_distraction_secs: float = 0.0
    events: list = field(default_factory=list)

    @property
    def duration_secs(self):
        return (self.end_time or time.time()) - self.start_time

    @property
    def focus_percentage(self):
        if self.duration_secs == 0:
            return 100.0
        return max(0.0, min(100.0,
            (self.duration_secs - self.total_distraction_secs) / self.duration_secs * 100))

    def to_dict(self):
        return {
            "duration_secs":          round(self.duration_secs, 1),
            "focus_percentage":       round(self.focus_percentage, 1),
            "total_distractions":     self.total_distractions,
            "gaze_away_count":        self.gaze_away_count,
            "focus_lost_count":       self.focus_lost_count,
            "total_distraction_secs": round(self.total_distraction_secs, 1),
            "events": [
                {"kind": e.kind,
                 "timestamp": round(e.timestamp - self.start_time, 1),
                 "detail": e.detail}
                for e in self.events
            ],
        }


# ── Tracker ───────────────────────────────────────────────────────────────────

class StudyTracker:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        model_path = os.path.expanduser("~/.cache/eyetrax/mediapipe/face_landmarker.task")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.session: Optional[SessionStats] = None

        self._distraction_start: Optional[float] = None
        self._blink_start: Optional[float] = None
        self._side_frames = 0
        self._last_side_event = 0.0
        self._last_dist_event = 0.0

        self._gaze_history: deque = deque(maxlen=300)
        self.last_raw: dict = {}

        self.on_event = None
        self.on_frame = None
        self.on_alert = None

    # compatibilidade com server.py
    class _FakeEstimator:
        model = True
    estimator = _FakeEstimator()

    def load_model(self) -> bool:
        return True

    # ── Sessão ────────────────────────────────────────────────────────────────

    def start_session(self):
        self.session = SessionStats()
        self._distraction_start = None
        self._blink_start = None
        self._side_frames = 0
        self._last_side_event = 0.0
        self._last_dist_event = 0.0
        self._gaze_history.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_session(self):
        self._running = False
        if self.session:
            self.session.end_time = time.time()
        if self._thread:
            self._thread.join(timeout=5)

    def get_stats(self):
        return self.session.to_dict() if self.session else {}

    def get_gaze_history(self):
        return list(self._gaze_history)

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                iris_ratio, head_yaw, blink, v_iris = self._extract(frame)
                now = time.time()

                is_distracted = self._detect(iris_ratio, v_iris, head_yaw, blink, now)

                if is_distracted:
                    if self._distraction_start is None:
                        self._distraction_start = now
                    elif now - self._distraction_start >= ALERT_DISTRACTION_SECS:
                        if self.on_alert:
                            self.on_alert("Você está distraído! Volte ao foco.")
                        self._distraction_start = now
                else:
                    if self._distraction_start is not None:
                        self.session.total_distraction_secs += now - self._distraction_start
                    self._distraction_start = None

                annotated = self._draw(frame.copy(), iris_ratio, head_yaw, blink, is_distracted)
                if self.on_frame:
                    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.on_frame(jpeg.tobytes())

                self.last_raw = {
                    "iris_ratio": round(iris_ratio, 3) if iris_ratio is not None else None,
                    "v_iris":     round(v_iris,     3) if v_iris     is not None else None,
                    "head_yaw":   round(head_yaw,   3) if head_yaw   is not None else None,
                    "blink":      blink,
                    "side_frames": self._side_frames,
                    "distracted":  is_distracted,
                }
                self._gaze_history.append({
                    "t": round(now - self.session.start_time, 1),
                    "iris_ratio": round(iris_ratio, 3) if iris_ratio is not None else None,
                    "distracted": is_distracted,
                })
        finally:
            cap.release()

    # ── Extração ──────────────────────────────────────────────────────────────

    def _extract(self, frame) -> Tuple:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None, None, False, None

        lm = result.face_landmarks[0]   # lista de NormalizedLandmark

        # ── íris ratio horizontal (média dos dois olhos) ──────────────────────
        def ratio(iris_i, left_i, right_i):
            w = lm[right_i].x - lm[left_i].x
            if w < 1e-4:
                return None
            return (lm[iris_i].x - lm[left_i].x) / w

        r_ratio = ratio(R_IRIS, R_EYE_LEFT, R_EYE_RIGHT)
        l_ratio = ratio(L_IRIS, L_EYE_LEFT, L_EYE_RIGHT)
        vals = [v for v in [r_ratio, l_ratio] if v is not None]
        iris_ratio = float(max(vals, key=lambda v: abs(v - 0.5))) if vals else None

        # ── íris ratio vertical (cima/baixo) ──────────────────────────────────
        def v_ratio(iris_i, top_i, bot_i):
            h = lm[bot_i].y - lm[top_i].y   # y cresce para baixo no MediaPipe
            if h < 1e-4:
                return None
            return (lm[iris_i].y - lm[top_i].y) / h

        v_vals = [v for v in [v_ratio(R_IRIS, R_EYE_TOP, R_EYE_BOTTOM),
                               v_ratio(L_IRIS, L_EYE_TOP, L_EYE_BOTTOM)] if v is not None]
        v_iris = float(max(v_vals, key=lambda v: abs(v - 0.5))) if v_vals else None

        # ── head yaw: nariz deslocado em relação ao centro dos olhos ─────────
        eye_cx    = (lm[R_EYE_LEFT].x + lm[L_EYE_RIGHT].x) / 2
        face_w    = abs(lm[L_EYE_RIGHT].x - lm[R_EYE_LEFT].x)
        head_yaw  = (lm[NOSE_TIP].x - eye_cx) / face_w if face_w > 1e-4 else None

        # ── EAR médio para detecção de piscar ─────────────────────────────────
        def ear(top, bot, left, right):
            v = abs(lm[top].y - lm[bot].y)
            h = abs(lm[left].x - lm[right].x)
            return v / h if h > 1e-4 else 1.0

        avg_ear = (ear(R_EYE_TOP, R_EYE_BOTTOM, R_EYE_LEFT, R_EYE_RIGHT) +
                   ear(L_EYE_TOP, L_EYE_BOTTOM, L_EYE_LEFT, L_EYE_RIGHT)) / 2
        blink = avg_ear < EAR_THRESHOLD

        return iris_ratio, head_yaw, blink, v_iris

    # ── Detecção ──────────────────────────────────────────────────────────────

    def _detect(self, iris_ratio, v_iris, head_yaw, blink, now) -> bool:
        distracted = False

        looking_side = False
        if iris_ratio is None and head_yaw is None:
            # sem face detectada — olhos fora do campo da câmera
            looking_side = True
        elif iris_ratio is not None:
            # horizontal: íris fora do centro
            looking_side = not (IRIS_SIDE_LOW <= iris_ratio <= IRIS_SIDE_HIGH)
            # vertical: olhando para cima ou para baixo
            if not looking_side and v_iris is not None:
                looking_side = not (IRIS_V_LOW <= v_iris <= IRIS_V_HIGH)
        elif head_yaw is not None and abs(head_yaw) > HEAD_YAW_THRESH:
            # fallback quando íris não está disponível mas face foi detectada
            looking_side = True

        self._side_frames = (self._side_frames + 1) if looking_side else max(0, self._side_frames - 1)

        if self._side_frames >= SMOOTH_FRAMES:
            distracted = True
            if now - self._last_side_event > DISTRACTION_COOLDOWN:
                self._last_side_event = now
                if iris_ratio is not None and not (IRIS_SIDE_LOW <= iris_ratio <= IRIS_SIDE_HIGH):
                    direction = "a esquerda" if iris_ratio < 0.5 else "a direita"
                elif v_iris is not None and not (IRIS_V_LOW <= v_iris <= IRIS_V_HIGH):
                    direction = "cima" if v_iris < 0.5 else "baixo"
                elif head_yaw is not None:
                    direction = "a esquerda" if head_yaw < 0 else "a direita"
                else:
                    direction = "fora da camera"
                self._register_event("side_gaze", now, f"olhando para {direction}")

        # olhos fechados por muito tempo
        if blink:
            if self._blink_start is None:
                self._blink_start = now
            elif now - self._blink_start >= BLINK_SUSTAINED_SECS:
                distracted = True
                self._register_event("focus_lost", now,
                                     f"olhos fechados por {now - self._blink_start:.1f}s")
                self._blink_start = now
        else:
            self._blink_start = None

        return distracted

    def _register_event(self, kind, ts, detail=""):
        if not self.session:
            return
        ev = Event(kind=kind, timestamp=ts, detail=detail)
        self.session.events.append(ev)
        if kind == "side_gaze":
            self.session.gaze_away_count += 1
            self.session.total_distractions += 1
        elif kind == "distraction":
            self.session.total_distractions += 1
        elif kind == "focus_lost":
            self.session.focus_lost_count += 1
            self.session.total_distractions += 1
        if self.on_event:
            self.on_event(ev, self.session.to_dict())

    # ── Overlay ───────────────────────────────────────────────────────────────

    def _draw(self, frame, iris_ratio, head_yaw, blink, distracted) -> np.ndarray:
        h, w = frame.shape[:2]

        color = (0, 200, 0) if not distracted else (0, 0, 220)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, "FOCADO" if not distracted else "DISTRAIDO",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        if blink:
            cv2.putText(frame, "PISCANDO", (w - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

        # barra de íris
        bx1, bx2, by = 20, w - 20, h - 20
        blen = bx2 - bx1
        cv2.rectangle(frame, (bx1, by - 6), (bx2, by + 6), (40, 40, 40), -1)
        cx1 = bx1 + int(IRIS_SIDE_LOW  * blen)
        cx2 = bx1 + int(IRIS_SIDE_HIGH * blen)
        cv2.rectangle(frame, (cx1, by - 6), (cx2, by + 6), (0, 60, 0), -1)

        if iris_ratio is not None:
            dx = bx1 + int(np.clip(iris_ratio, 0, 1) * blen)
            dc = (0, 220, 0) if IRIS_SIDE_LOW <= iris_ratio <= IRIS_SIDE_HIGH else (0, 0, 220)
            cv2.circle(frame, (dx, by), 9, dc, -1)
            cv2.putText(frame, f"iris={iris_ratio:.2f}  yaw={head_yaw:+.2f}" if head_yaw else f"iris={iris_ratio:.2f}",
                        (bx1, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        return frame
