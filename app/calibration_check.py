"""
Tela de verificação pós-calibração.
Mostra o ponto de gaze ao vivo para o usuário confirmar se está correto.
Pressione ENTER para aceitar ou R para recalibrar.
"""
import cv2
import numpy as np
import time
from collections import deque

from eyetrax import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


def run_calibration_check(estimator: GazeEstimator, camera_index: int = 0) -> bool:
    """
    Abre uma janela fullscreen mostrando o ponto de gaze em tempo real.
    Retorna True se o usuário aceitar, False se quiser recalibrar.
    """
    sw, sh = get_screen_size()
    cap = open_camera(camera_index)

    WIN = "Verificar Calibração"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # histórico para suavizar o ponto
    history = deque(maxlen=8)

    # pontos de referência nos cantos + centro
    ref_points = [
        (int(sw * 0.1),  int(sh * 0.1)),
        (int(sw * 0.9),  int(sh * 0.1)),
        (int(sw * 0.5),  int(sh * 0.5)),
        (int(sw * 0.1),  int(sh * 0.9)),
        (int(sw * 0.9),  int(sh * 0.9)),
    ]

    result = True
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        features, blink = estimator.extract_features(frame)

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (15, 17, 23)  # fundo escuro

        # ── instruções ────────────────────────────────────────────────────
        msg1 = "Olhe ao redor da tela — o circulo azul deve seguir seu olhar"
        msg2 = "ENTER = calibracao OK    |    R = refazer calibracao    |    ESC = cancelar"
        cv2.putText(canvas, msg1, (sw//2 - 420, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(canvas, msg2, (sw//2 - 440, sh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        # ── pontos de referência (cruzes brancas) ─────────────────────────
        for px, py in ref_points:
            cv2.drawMarker(canvas, (px, py), (80, 80, 80),
                           cv2.MARKER_CROSS, 30, 1)

        # ── ponto de gaze ─────────────────────────────────────────────────
        if features is not None and not blink:
            try:
                gx, gy = estimator.predict([features])[0]
                history.append((gx, gy))
            except Exception:
                pass

        if len(history) >= 3:
            avg_x = np.mean([p[0] for p in history])
            avg_y = np.mean([p[1] for p in history])
            px = int(np.clip(avg_x, 0, 1) * sw)
            py = int(np.clip(avg_y, 0, 1) * sh)

            # sombra
            cv2.circle(canvas, (px, py), 28, (20, 60, 120), -1)
            # círculo externo
            cv2.circle(canvas, (px, py), 24, (30, 144, 255), 2)
            # ponto central
            cv2.circle(canvas, (px, py), 6,  (30, 144, 255), -1)

            # indicador de qualidade (quanto do histórico está dentro da tela)
            on_screen = sum(
                1 for gx, gy in history
                if -0.1 <= gx <= 1.1 and -0.1 <= gy <= 1.1
            )
            quality = on_screen / len(history)
            q_color = (0, 200, 80) if quality > 0.7 else (0, 160, 255) if quality > 0.4 else (50, 50, 220)
            label = "Rastreando" if quality > 0.7 else ("Parcial" if quality > 0.4 else "Fora da tela")
            cv2.putText(canvas, label, (px + 30, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, 1)
        else:
            # sem detecção
            cv2.putText(canvas, "Rosto nao detectado", (sw//2 - 160, sh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)

        # miniatura da câmera (canto inferior direito)
        thumb_w, thumb_h = 200, 150
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        canvas[sh - thumb_h - 10: sh - 10, sw - thumb_w - 10: sw - 10] = thumb

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord(" "):   # ENTER ou ESPAÇO = aceitar
            result = True
            break
        elif key == ord("r") or key == ord("R"):
            result = False
            break
        elif key == 27:                    # ESC = cancelar
            result = True
            break

    cap.release()
    cv2.destroyWindow(WIN)
    return result
