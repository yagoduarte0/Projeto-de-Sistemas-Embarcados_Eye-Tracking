"""
Study Tracker — sem calibração necessária.

Uso:
  python main.py
  python main.py --port 8080
"""
import argparse
import threading
import time

def main():
    parser = argparse.ArgumentParser(description="Study Tracker")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    from app.server import tracker, run as run_server

    # Flask em background
    threading.Thread(target=run_server, kwargs={"port": args.port}, daemon=True).start()
    print(f"[Servidor] http://127.0.0.1:{args.port}")
    time.sleep(1.0)

    # Overlay na thread principal
    from app.overlay import StudyOverlay
    StudyOverlay(tracker, port=args.port).start()

if __name__ == "__main__":
    main()
