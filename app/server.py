"""
Servidor Flask + SocketIO para o dashboard de estudos.
"""
import os
import time
from flask import Flask, render_template, send_file, jsonify, request
from flask_socketio import SocketIO
import io

from .tracker import StudyTracker
from .reports import export_csv, export_pdf

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)
app.config["SECRET_KEY"] = "study-tracker-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

tracker = StudyTracker()


# ── Callbacks do tracker ──────────────────────────────────────────────────────

def on_event(event, stats):
    socketio.emit("stats_update", stats)
    socketio.emit("new_event", {
        "kind": event.kind,
        "timestamp": round(event.timestamp - tracker.session.start_time, 1),
        "detail": event.detail,
    })


def on_frame(jpeg_bytes):
    import base64
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    socketio.emit("frame", {"data": b64})


def on_alert(message):
    socketio.emit("alert", {"message": message})


tracker.on_event = on_event
tracker.on_frame = on_frame
tracker.on_alert = on_alert


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results")
def results():
    stats = tracker.get_stats()
    return render_template("results.html", stats=stats)


@app.route("/api/status")
def status():
    return jsonify({
        "running": tracker._running,
        "model_loaded": tracker.estimator.model is not None,
        "stats": tracker.get_stats() if tracker.session else {},
    })


@app.route("/api/start", methods=["POST"])
def start_session():
    if tracker._running:
        return jsonify({"error": "Sessão já em andamento"}), 400
    tracker.start_session()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_session():
    tracker.stop_session()
    return jsonify({"ok": True, "stats": tracker.get_stats()})


@app.route("/api/stats")
def get_stats():
    return jsonify(tracker.get_stats())


@app.route("/api/gaze_history")
def gaze_history():
    return jsonify(tracker.get_gaze_history())


@app.route("/api/debug")
def debug():
    return jsonify(tracker.last_raw)


@app.route("/api/export/csv")
def export_csv_route():
    stats = tracker.get_stats()
    if not stats:
        return jsonify({"error": "Sem dados de sessão"}), 400
    data = export_csv(stats)
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"sessao_{time.strftime('%Y%m%d_%H%M%S')}.csv",
    )


@app.route("/api/export/pdf")
def export_pdf_route():
    stats = tracker.get_stats()
    if not stats:
        return jsonify({"error": "Sem dados de sessão"}), 400
    data = export_pdf(stats)
    return send_file(
        io.BytesIO(data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"sessao_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
    )


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    if tracker.session:
        socketio.emit("stats_update", tracker.get_stats())


def run(host="127.0.0.1", port=5000, debug=False):
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
