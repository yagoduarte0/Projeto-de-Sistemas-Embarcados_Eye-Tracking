"""
Rastreamento de olhar usando MediaPipe FaceLandmarker (Tasks API).
Sem calibração. Funciona direto da câmera.

Pipeline de sinais:
  - Filtro de Kalman 1D aplicado ao sinal de posição da íris (H e V)
  - IAF (Índice de Atenção Focalizada): fusão ponderada de íris, yaw e EAR
  - Detecção de distração via sinal filtrado + smoothing temporal
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

# ── Thresholds de detecção ────────────────────────────────────────────────────
IRIS_SIDE_LOW   = 0.33   # ratio < → olhando para esquerda
IRIS_SIDE_HIGH  = 0.67   # ratio > → olhando para direita
IRIS_V_LOW      = 0.25   # ratio vertical < → olhando para cima
IRIS_V_HIGH     = 0.75   # ratio vertical > → olhando para baixo
HEAD_YAW_THRESH = 0.18   # desvio normalizado do nariz
EAR_THRESHOLD   = 0.22   # EAR abaixo = olho fechado (inclui transição do piscar)
BLINK_SUSTAINED_SECS   = 2.5
DISTRACTION_COOLDOWN   = 3.0
ALERT_DISTRACTION_SECS = 5.0
SMOOTH_FRAMES          = 5    # frames consecutivos para confirmar estado

# ── Pesos do IAF (soma = 1.0) ─────────────────────────────────────────────────
W_IRIS_H = 0.45   # desvio horizontal da íris (sinal mais discriminativo)
W_IRIS_V = 0.20   # desvio vertical da íris
W_YAW    = 0.20   # rotação horizontal da cabeça
W_EAR    = 0.15   # fechamento dos olhos (sonolência)


# ── Filtro de Kalman 1D ───────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Filtro de Kalman escalar para suavização do sinal de posição da íris.

    Modelo de processo:  x_k = x_{k-1} + w_k     (random walk)
    Modelo de medição:   z_k = x_k    + v_k

    Parâmetros:
        q — ruído de processo  (quão rápido o sinal pode mudar entre frames)
        r — ruído de medição   (imprecisão dos landmarks MediaPipe)
        x0 — estimativa inicial do estado
    """
    def __init__(self, q: float = 1e-5, r: float = 5e-3, x0: float = 0.5):
        self._q = q
        self._r = r
        self._p = 1.0   # covariância do erro (alta = incerto)
        self._x = x0    # estimativa inicial

    def update(self, z: float) -> float:
        """Incorpora nova medição z e retorna a estimativa filtrada."""
        # Predição: aumenta incerteza
        self._p += self._q
        # Ganho de Kalman: pondera confiança entre predição e medição
        k = self._p / (self._p + self._r)
        # Atualização: corrige estimativa com a inovação (z - x)
        self._x += k * (z - self._x)
        self._p *= (1.0 - k)
        return float(self._x)

    def predict(self) -> float:
        """Propaga o estado sem medição (frame sem rosto detectado)."""
        self._p += self._q
        return float(self._x)

    def reset(self, x0: float = 0.5) -> None:
        self._p = 1.0
        self._x = x0


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
    iaf_sum: float = 0.0      # acumulador para média do IAF
    iaf_count: int = 0
    iaf_min: float = 1.0      # pior momento de atenção da sessão
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

    @property
    def iaf_mean(self) -> float:
        return self.iaf_sum / self.iaf_count if self.iaf_count > 0 else 1.0

    def to_dict(self):
        return {
            "duration_secs":          round(self.duration_secs, 1),
            "focus_percentage":       round(self.focus_percentage, 1),
            "total_distractions":     self.total_distractions,
            "gaze_away_count":        self.gaze_away_count,
            "focus_lost_count":       self.focus_lost_count,
            "total_distraction_secs": round(self.total_distraction_secs, 1),
            "iaf_mean":               round(self.iaf_mean * 100, 1),
            "iaf_min":                round(self.iaf_min * 100, 1),
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

        # Filtros de Kalman independentes para componente H e V da íris
        self._kalman_iris = KalmanFilter1D(q=1e-5, r=5e-3, x0=0.5)
        self._kalman_v    = KalmanFilter1D(q=1e-5, r=5e-3, x0=0.5)

        # EAR suavizado: decai lentamente durante fechamento prolongado,
        # recupera rápido quando olho abre — piscadas normais não afetam o IAF
        self._ear_smooth: float = 1.0

        # Parâmetros de calibração (carregados de arquivo pela calibração de 5 pontos)
        self._iris_center_h: float = 0.5
        self._iris_center_v: float = 0.5
        self._iris_flat_h: float = 0.05
        self._iris_flat_v: float = 0.07

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
        self.on_frame = None   # assinatura: on_frame(jpeg_bytes: bytes, iaf: float)
        self.on_alert = None

        # Câmera aberta uma única vez — compartilhada entre calibração e sessões.
        # Evita o delay de 3-7s do DirectShow a cada abertura no Windows.
        self._cap = cv2.VideoCapture(self.camera_index)

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
        self._kalman_iris.reset()
        self._kalman_v.reset()
        self._ear_smooth = 1.0
        self._load_calibration()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_session(self):
        self._running = False
        if self.session:
            self.session.end_time = time.time()
        if self._thread:
            self._thread.join(timeout=5)

    def _load_calibration(self):
        from app.calibration_check import load_calibration
        calib = load_calibration()
        if calib:
            self._iris_center_h = calib["iris_center_h"]
            self._iris_center_v = calib["iris_center_v"]
            self._iris_flat_h   = calib["iris_flat_h"]
            self._iris_flat_v   = calib["iris_flat_v"]
        else:
            self._iris_center_h = 0.5
            self._iris_center_v = 0.5
            self._iris_flat_h   = 0.05
            self._iris_flat_v   = 0.07

    def get_stats(self):
        return self.session.to_dict() if self.session else {}

    def get_gaze_history(self):
        return list(self._gaze_history)

    # ── Índice de Atenção Focalizada (IAF) ────────────────────────────────────

    def _compute_iaf(
        self,
        iris_filt: Optional[float],
        v_filt: Optional[float],
        head_yaw: Optional[float],
        ear_smooth: float,
    ) -> float:
        """
        Computa o IAF em [0.0, 1.0] via fusão ponderada de quatro componentes:

          iris_h    — proximidade horizontal da íris ao centro (0.5)
          iris_v    — proximidade vertical da íris ao centro
          yaw       — alinhamento da cabeça (sem rotação lateral)
          ear_smooth — fechamento ocular suavizado (imune a piscadas normais)

        Sem detecção de rosto → IAF = 0 (ausência total de atenção).
        """
        if iris_filt is None:
            return 0.0

        # Componente horizontal: dead zone adaptativa ao tamanho da tela.
        # Dentro do raio _iris_flat_h (calibrado nos primeiros 3s) → iris_h = 1.0.
        # Além dele, penalidade linear até o threshold de detecção (0.17).
        excess_h = max(0.0, abs(iris_filt - self._iris_center_h) - self._iris_flat_h)
        dev_h = excess_h / (0.5 - IRIS_SIDE_LOW)
        iris_h = float(max(0.0, 1.0 - dev_h))

        # Componente vertical: mesma lógica com raio _iris_flat_v calibrado
        if v_filt is not None:
            excess_v = max(0.0, abs(v_filt - self._iris_center_v) - self._iris_flat_v)
            dev_v = excess_v / (0.5 - IRIS_V_LOW)
            iris_v = float(max(0.0, 1.0 - dev_v))
        else:
            iris_v = iris_h   # fallback: usa componente horizontal

        # Componente yaw: normalizado pelo threshold de rotação
        if head_yaw is not None:
            yaw_score = float(max(0.0, 1.0 - abs(head_yaw) / HEAD_YAW_THRESH))
        else:
            yaw_score = 1.0   # sem detecção de yaw = sem penalidade

        # Componente EAR: suavizado — piscadas normais mal afetam o IAF
        ear_score = ear_smooth

        iaf = W_IRIS_H * iris_h + W_IRIS_V * iris_v + W_YAW * yaw_score + W_EAR * ear_score
        return round(float(np.clip(iaf, 0.0, 1.0)), 3)

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(self):
        if not self._cap.isOpened():
            return
        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                iris_raw, head_yaw, blink, v_raw = self._extract(frame)
                now = time.time()

                # ── Filtro de Kalman ───────────────────────────────────────────
                if iris_raw is not None:
                    if blink:
                        # Durante piscada os landmarks de íris são imprecisos —
                        # apenas propaga o estado sem corrigir para não contaminar
                        # iris_filt com leituras ruins. Quando o olho abrir,
                        # o filtro retoma a partir da última estimativa válida.
                        iris_filt = self._kalman_iris.predict()
                        v_filt    = self._kalman_v.predict()
                    else:
                        iris_filt = self._kalman_iris.update(iris_raw)
                        v_filt    = self._kalman_v.update(v_raw) if v_raw is not None \
                                    else self._kalman_v.predict()
                else:
                    iris_filt = None
                    v_filt    = None
                    self._kalman_iris.predict()
                    self._kalman_v.predict()

                # ── EAR suavizado ─────────────────────────────────────────────
                # Decai 0.5/s durante fechamento → piscada de 200ms cai ~10%
                # Recupera 8.0/s quando olho abre   → volta em ~125ms
                dt = now - self._last_frame_time if hasattr(self, '_last_frame_time') else 0.033
                self._last_frame_time = now
                if blink:
                    self._ear_smooth = max(0.0, self._ear_smooth - dt * 0.5)
                else:
                    self._ear_smooth = min(1.0, self._ear_smooth + dt * 8.0)

                # ── IAF ────────────────────────────────────────────────────────
                iaf = self._compute_iaf(iris_filt, v_filt, head_yaw, self._ear_smooth)

                self.session.iaf_sum   += iaf
                self.session.iaf_count += 1
                if iaf < self.session.iaf_min:
                    self.session.iaf_min = iaf

                # ── Detecção (opera sobre sinal filtrado) ──────────────────────
                is_distracted = self._detect(iris_filt, v_filt, head_yaw, blink, now)

                if is_distracted:
                    if self._distraction_start is None:
                        self._distraction_start = now
                    elif now - self._distraction_start >= ALERT_DISTRACTION_SECS:
                        self._register_event(
                            "focus_lost", now,
                            f"distração contínua por {ALERT_DISTRACTION_SECS:.0f}s"
                        )
                        if self.on_alert:
                            self.on_alert("Você está distraído! Volte ao foco.")
                        self._distraction_start = now
                else:
                    if self._distraction_start is not None:
                        self.session.total_distraction_secs += now - self._distraction_start
                    self._distraction_start = None

                annotated = self._draw(frame.copy(), iris_filt, head_yaw, blink,
                                       is_distracted, iaf)
                if self.on_frame:
                    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.on_frame(jpeg.tobytes(), iaf)

                self.last_raw = {
                    "iris_raw":       round(iris_raw,  3) if iris_raw  is not None else None,
                    "iris_filt":      round(iris_filt, 3) if iris_filt is not None else None,
                    "v_raw":          round(v_raw,     3) if v_raw     is not None else None,
                    "v_filt":         round(v_filt,    3) if v_filt    is not None else None,
                    "head_yaw":       round(head_yaw,  3) if head_yaw  is not None else None,
                    "blink":          blink,
                    "side_frames":    self._side_frames,
                    "distracted":     is_distracted,
                    "iaf":            iaf,
                    "calib_center_h": round(self._iris_center_h, 3),
                    "calib_center_v": round(self._iris_center_v, 3),
                    "calib_flat_h":   round(self._iris_flat_h, 3),
                    "calib_flat_v":   round(self._iris_flat_v, 3),
                    "calibrated":     self._iris_center_h != 0.5 or self._iris_flat_h != 0.05,
                }
                self._gaze_history.append({
                    "t":          round(now - self.session.start_time, 1),
                    "iris_raw":   round(iris_raw,  3) if iris_raw  is not None else None,
                    "iris_filt":  round(iris_filt, 3) if iris_filt is not None else None,
                    "iaf":        iaf,
                    "distracted": is_distracted,
                })
        except Exception:
            import traceback; traceback.print_exc()

    # ── Extração de features ──────────────────────────────────────────────────

    def _extract(self, frame) -> Tuple:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None, None, False, None

        lm = result.face_landmarks[0]

        # íris ratio horizontal (média ponderada dos dois olhos)
        def ratio(iris_i, left_i, right_i):
            w = lm[right_i].x - lm[left_i].x
            if w < 1e-4:
                return None
            return (lm[iris_i].x - lm[left_i].x) / w

        r_ratio = ratio(R_IRIS, R_EYE_LEFT, R_EYE_RIGHT)
        l_ratio = ratio(L_IRIS, L_EYE_LEFT, L_EYE_RIGHT)
        vals = [v for v in [r_ratio, l_ratio] if v is not None]
        iris_ratio = float(max(vals, key=lambda v: abs(v - 0.5))) if vals else None

        # íris ratio vertical
        def v_ratio(iris_i, top_i, bot_i):
            h = lm[bot_i].y - lm[top_i].y
            if h < 1e-4:
                return None
            return (lm[iris_i].y - lm[top_i].y) / h

        v_vals = [v for v in [v_ratio(R_IRIS, R_EYE_TOP, R_EYE_BOTTOM),
                               v_ratio(L_IRIS, L_EYE_TOP, L_EYE_BOTTOM)] if v is not None]
        v_iris = float(max(v_vals, key=lambda v: abs(v - 0.5))) if v_vals else None

        # head yaw: desvio do nariz em relação ao centro dos olhos
        eye_cx   = (lm[R_EYE_LEFT].x + lm[L_EYE_RIGHT].x) / 2
        face_w   = abs(lm[L_EYE_RIGHT].x - lm[R_EYE_LEFT].x)
        head_yaw = (lm[NOSE_TIP].x - eye_cx) / face_w if face_w > 1e-4 else None

        # EAR (Eye Aspect Ratio) para detecção de piscar
        def ear(top, bot, left, right):
            v = abs(lm[top].y - lm[bot].y)
            h = abs(lm[left].x - lm[right].x)
            return v / h if h > 1e-4 else 1.0

        avg_ear = (ear(R_EYE_TOP, R_EYE_BOTTOM, R_EYE_LEFT, R_EYE_RIGHT) +
                   ear(L_EYE_TOP, L_EYE_BOTTOM, L_EYE_LEFT, L_EYE_RIGHT)) / 2
        blink = avg_ear < EAR_THRESHOLD

        return iris_ratio, head_yaw, blink, v_iris

    # ── Detecção de distração (opera sobre sinal Kalman-filtrado) ─────────────

    def _detect(self, iris_filt, v_filt, head_yaw, blink, now) -> bool:
        distracted = False

        looking_side = False
        if iris_filt is None and head_yaw is None:
            looking_side = True   # sem face = fora de câmera

        elif blink:
            # Durante piscada os landmarks de íris são numericamente instáveis
            # (denominador do v_ratio → 0 com pálpebras fechadas).
            # Usa apenas head_yaw para não penalizar piscadas normais.
            looking_side = head_yaw is not None and abs(head_yaw) > HEAD_YAW_THRESH

        elif iris_filt is not None:
            looking_side_h = not (IRIS_SIDE_LOW <= iris_filt <= IRIS_SIDE_HIGH)

            # Compensação oculomotora (reflexo vestíbulo-ocular):
            # ao girar a cabeça para um lado, a íris vai naturalmente para o lado
            # oposto para manter a fixação na tela — não é olhar evasivo.
            # Padrão: cabeça direita (yaw > 0) → íris esquerda (ratio < 0.5) → OK
            #         cabeça esquerda (yaw < 0) → íris direita (ratio > 0.5) → OK
            if looking_side_h and head_yaw is not None:
                vor = (head_yaw > 0 and iris_filt < IRIS_SIDE_LOW) or \
                      (head_yaw < 0 and iris_filt > IRIS_SIDE_HIGH)
                if vor:
                    looking_side_h = False

            looking_side = looking_side_h

        elif head_yaw is not None and abs(head_yaw) > HEAD_YAW_THRESH:
            looking_side = True   # fallback: apenas yaw disponível

        # Sobe até SMOOTH_FRAMES (sem acumular além disso) e desce 2x mais rápido
        if looking_side:
            self._side_frames = min(self._side_frames + 1, SMOOTH_FRAMES)
        else:
            self._side_frames = max(0, self._side_frames - 2)

        if self._side_frames >= SMOOTH_FRAMES:
            distracted = True
            if now - self._last_side_event > DISTRACTION_COOLDOWN:
                self._last_side_event = now
                if iris_filt is not None and not (IRIS_SIDE_LOW <= iris_filt <= IRIS_SIDE_HIGH):
                    direction = "a esquerda" if iris_filt < 0.5 else "a direita"
                elif v_filt is not None and not (IRIS_V_LOW <= v_filt <= IRIS_V_HIGH):
                    direction = "cima" if v_filt < 0.5 else "baixo"
                elif head_yaw is not None:
                    direction = "a esquerda" if head_yaw < 0 else "a direita"
                else:
                    direction = "fora da camera"
                self._register_event("side_gaze", now, f"olhando para {direction}")

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

    # ── Frame anotado ─────────────────────────────────────────────────────────

    def _draw(self, frame, iris_filt, head_yaw, blink, distracted, iaf) -> np.ndarray:
        h, w = frame.shape[:2]

        # barra de status no topo
        status_color = (0, 200, 0) if not distracted else (0, 0, 220)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, "FOCADO" if not distracted else "DISTRAIDO",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        # IAF no canto superior direito
        iaf_pct = int(iaf * 100)
        iaf_color = (0, 200, 0) if iaf >= 0.7 else (0, 165, 255) if iaf >= 0.4 else (0, 0, 220)
        cv2.putText(frame, f"IAF {iaf_pct}%",
                    (w - 115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, iaf_color, 2)

        if blink:
            cv2.putText(frame, "PISCANDO", (w - 175, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

        # barra de posição da íris (valor filtrado pelo Kalman)
        bx1, bx2, by = 20, w - 20, h - 20
        blen = bx2 - bx1
        cv2.rectangle(frame, (bx1, by - 6), (bx2, by + 6), (40, 40, 40), -1)
        cx1 = bx1 + int(IRIS_SIDE_LOW  * blen)
        cx2 = bx1 + int(IRIS_SIDE_HIGH * blen)
        cv2.rectangle(frame, (cx1, by - 6), (cx2, by + 6), (0, 60, 0), -1)

        if iris_filt is not None:
            dx = bx1 + int(np.clip(iris_filt, 0, 1) * blen)
            dc = (0, 220, 0) if IRIS_SIDE_LOW <= iris_filt <= IRIS_SIDE_HIGH else (0, 0, 220)
            cv2.circle(frame, (dx, by), 9, dc, -1)
            label = f"iris_k={iris_filt:.2f}"
            if head_yaw is not None:
                label += f"  yaw={head_yaw:+.2f}"
            cv2.putText(frame, label,
                        (bx1, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        return frame
