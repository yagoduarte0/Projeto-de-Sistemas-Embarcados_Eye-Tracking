"""
Calibração de 5 pontos do IAF (Índice de Atenção Focalizada).

Uso normal (via overlay): CalibrationWindow(parent_root, tracker, on_done)
Uso standalone (--calibrate): run_calibration(tracker)

Salva resultado em ~/.cache/eyetrax/study_tracker_calib.json.
"""
import cv2
import json
import time
import threading
import tkinter as tk
import numpy as np
from pathlib import Path

CALIB_FILE = Path.home() / ".cache" / "eyetrax" / "study_tracker_calib.json"

# 0=centro  1=topo-esq  2=topo-dir  3=baixo-esq  4=baixo-dir
CALIB_POINTS = [
    (0.50, 0.50),
    (0.15, 0.15),
    (0.85, 0.15),
    (0.15, 0.85),
    (0.85, 0.85),
]
DWELL_SECS  = 2.5
SETTLE_SECS = 0.7

BG      = "#0f1117"
BLUE    = "#1e90ff"
GREEN   = "#00c896"
MUTED   = "#7b82a8"
WHITE   = "#e8eaf6"
SURFACE = "#1a1d27"


class CalibrationWindow:
    """
    Janela de calibração como Toplevel dentro de um Tk existente.
    Não bloqueia — usa after() para animação e thread para câmera.
    Chama on_done(success: bool) quando terminar ou cancelar.
    """

    def __init__(self, parent: tk.Tk, tracker, on_done):
        self.tracker  = tracker
        self.on_done  = on_done

        self._state = {
            "point_idx":  0,
            "collecting": False,
            "done":       False,
            "cancelled":  False,
        }
        self._samples: list[list] = [[] for _ in CALIB_POINTS]
        self._point_start = time.time()

        self._top = tk.Toplevel(parent)
        self._top.attributes("-topmost", True)
        self._top.overrideredirect(True)          # sem barra de título
        self._top.configure(bg=BG)
        self._top.bind("<Escape>", self._on_escape)

        self._sw = self._top.winfo_screenwidth()
        self._sh = self._top.winfo_screenheight()
        # Posiciona manualmente cobrindo a tela toda (confiável no Windows)
        self._top.geometry(f"{self._sw}x{self._sh}+0+0")

        self._cv = tk.Canvas(self._top, bg=BG, highlightthickness=0)
        self._cv.pack(fill="both", expand=True)

        # Reutiliza a câmera já aberta do tracker — sem delay de inicialização
        self._cap = tracker._cap
        tracker._kalman_iris.reset()
        tracker._kalman_v.reset()
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self._top.after(150, self._draw)

    # ── Thread da câmera ──────────────────────────────────────────────────────

    def _camera_loop(self):
        while not self._state["done"]:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            iris_raw, _, blink, v_raw, avg_ear = self.tracker._extract(frame)
            if iris_raw is not None:
                if avg_ear < 0.15:
                    self.tracker._kalman_iris.predict()
                    self.tracker._kalman_v.predict()
                else:
                    ih = self.tracker._kalman_iris.update(iris_raw)
                    iv = self.tracker._kalman_v.update(v_raw) if v_raw is not None \
                         else self.tracker._kalman_v.predict()
                    if self._state["collecting"]:
                        idx = self._state["point_idx"]
                        if idx < len(CALIB_POINTS):
                            self._samples[idx].append((float(ih), float(iv)))
            else:
                self.tracker._kalman_iris.predict()
                self.tracker._kalman_v.predict()
        # Não libera — câmera pertence ao tracker e será reutilizada na sessão

    # ── Animação (main thread via after) ─────────────────────────────────────

    def _draw(self):
        if not self._top.winfo_exists() or self._state["done"]:
            return

        idx     = self._state["point_idx"]
        elapsed = time.time() - self._point_start
        sw, sh  = self._sw, self._sh
        cv      = self._cv

        self._state["collecting"] = elapsed > SETTLE_SECS

        cv.delete("all")
        cv.create_rectangle(0, 0, sw, sh, fill=BG, outline="")

        cv.create_text(sw // 2, 44,
                       text="Olhe fixamente para o círculo até a barra encher",
                       fill=WHITE, font=("Segoe UI", 18, "bold"))
        cv.create_text(sw // 2, 80,
                       text=f"Ponto {idx + 1} de {len(CALIB_POINTS)}  —  ESC para cancelar",
                       fill=MUTED, font=("Segoe UI", 13))

        # Barra de progresso
        bw, bh = 420, 12
        bx = sw // 2 - bw // 2
        by = sh - 60
        prog      = min(1.0, elapsed / DWELL_SECS)
        bar_color = GREEN if self._state["collecting"] else BLUE
        cv.create_rectangle(bx, by, bx + bw, by + bh, fill=SURFACE, outline="")
        cv.create_rectangle(bx, by, bx + int(bw * prog), by + bh, fill=bar_color, outline="")

        # Alvo
        fx, fy = CALIB_POINTS[idx]
        px, py = int(fx * sw), int(fy * sh)
        r = 22
        dot_color = GREEN if self._state["collecting"] else BLUE
        cv.create_oval(px - r - 6, py - r - 6, px + r + 6, py + r + 6,
                       outline=dot_color, width=2)
        cv.create_oval(px - r, py - r, px + r, py + r, fill=dot_color, outline="")
        cv.create_oval(px - 5, py - 5, px + 5, py + 5, fill=WHITE, outline="")

        if elapsed >= DWELL_SECS:
            self._state["point_idx"] += 1
            self._point_start = time.time()
            self._state["collecting"] = False

            if self._state["point_idx"] >= len(CALIB_POINTS):
                self._state["done"] = True
                self._top.after(300, self._finish)
                return

        self._top.after(33, self._draw)

    # ── Finalização ───────────────────────────────────────────────────────────

    def _finish(self):
        self._state["done"] = True
        self._cam_thread.join(timeout=3.0)   # garante que ninguém mais lê a câmera

        collected = {i: self._samples[i] for i in range(len(CALIB_POINTS))}
        result    = _compute(collected)

        success = False
        if result is not None:
            CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALIB_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            success = True

        if self._top.winfo_exists():
            self._top.destroy()
        self.on_done(success)

    def _on_escape(self, _event=None):
        self._state["done"]      = True
        self._state["cancelled"] = True
        self._cam_thread.join(timeout=3.0)
        if self._top.winfo_exists():
            self._top.destroy()
        self.on_done(False)


# ── Standalone (main.py --calibrate) ─────────────────────────────────────────

def run_calibration(tracker) -> bool:
    """
    Modo standalone: cria janela Tk própria, bloqueia até terminar.
    Usar apenas fora do loop do overlay (ex: main.py --calibrate).
    """
    result = [False]

    def on_done(success: bool):
        result[0] = success
        root.quit()

    root = tk.Tk()
    root.withdraw()   # esconde janela vazia — só o Toplevel é visível
    CalibrationWindow(root, tracker, on_done)
    root.mainloop()
    root.destroy()
    return result[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute(collected: dict) -> dict | None:
    def mh(indices):
        vals = [s[0] for i in indices for s in collected.get(i, [])]
        return float(np.mean(vals)) if vals else None

    def mv(indices):
        vals = [s[1] for i in indices for s in collected.get(i, [])]
        return float(np.mean(vals)) if vals else None

    left_h  = mh([1, 3])
    right_h = mh([2, 4])
    top_v   = mv([1, 2])
    bot_v   = mv([3, 4])
    cen_h   = mh([0])
    cen_v   = mv([0])

    if any(x is None for x in [left_h, right_h, top_v, bot_v, cen_h, cen_v]):
        return None

    center_h = (left_h + right_h) / 2
    center_v = (top_v  + bot_v)  / 2
    flat_h   = min(abs(center_h - left_h), abs(right_h - center_h)) * 0.85
    flat_v   = min(abs(center_v - top_v),  abs(bot_v  - center_v)) * 0.85
    flat_h   = max(flat_h, 0.03)
    flat_v   = max(flat_v, 0.04)

    return {
        "iris_center_h": round(center_h, 4),
        "iris_center_v": round(center_v, 4),
        "iris_flat_h":   round(flat_h,   4),
        "iris_flat_v":   round(flat_v,   4),
    }


def load_calibration() -> dict | None:
    if not CALIB_FILE.exists():
        return None
    try:
        with open(CALIB_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
