"""
Study Tracker

Uso:
  python main.py                  # roda normalmente (calibra se necessário)
  python main.py --calibrate      # força recalibração
  python main.py --port 8080
"""
import argparse
import threading
import time

def main():
    parser = argparse.ArgumentParser(description="Study Tracker")
    parser.add_argument("--port",      type=int,  default=5000)
    parser.add_argument("--calibrate", action="store_true",
                        help="Forçar recalibração antes de iniciar")
    args = parser.parse_args()

    from app.server import tracker, run as run_server

    # Flask em background
    threading.Thread(target=run_server, kwargs={"port": args.port}, daemon=True).start()
    print(f"[Servidor] http://127.0.0.1:{args.port}")
    time.sleep(1.0)

    # Calibração forçada via flag (opcional)
    if args.calibrate:
        from app.calibration_check import run_calibration
        print("[Calibração] Iniciando calibração de 5 pontos...")
        if run_calibration(tracker):
            tracker._load_calibration()
            print("[Calibração] Concluída.")
        else:
            print("[Calibração] Cancelada.")

    # Overlay na thread principal
    from app.overlay import StudyOverlay
    StudyOverlay(tracker, port=args.port).start()

if __name__ == "__main__":
    main()
