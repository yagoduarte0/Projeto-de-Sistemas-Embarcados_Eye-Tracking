"""
Overlay tkinter — janela sempre no topo mostrando métricas em tempo real.
Compacto durante a sessão; expande para resumo completo ao parar.
"""
import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import webbrowser

# cores
BG         = "#0f1117"
BG_CARD    = "#1a1d27"
BORDER     = "#2c3150"
TEXT       = "#e8eaf6"
MUTED      = "#7b82a8"
PRIMARY    = "#1e90ff"
SUCCESS    = "#00c896"
DANGER     = "#ff4d6d"
WARNING    = "#ffb347"
YELLOW     = "#ffd166"


class StudyOverlay:
    def __init__(self, tracker, port: int = 5000):
        self.tracker = tracker
        self.port = port
        self.root = None
        self._drag_x = 0
        self._drag_y = 0
        self._mode = "idle"   # idle | running | summary
        self._timer_job = None

    # ── Iniciar ───────────────────────────────────────────────────────────────

    def start(self):
        """Chama na thread principal."""
        self.root = tk.Tk()
        self.root.title("Study Tracker")
        self.root.configure(bg=BG)
        self.root.overrideredirect(True)       # sem barra de título
        self.root.attributes("-topmost", True) # sempre na frente
        self.root.attributes("-alpha", 0.93)

        self._build_compact()
        self._center_top_right()
        self._bind_drag()

        self.root.mainloop()

    # ── Layout compacto (idle / running) ──────────────────────────────────────

    def _build_compact(self):
        self._clear()
        self._mode = "idle" if not self.tracker._running else "running"

        root = self.root
        root.geometry("300x220")

        outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        inner = tk.Frame(outer, bg=BG, padx=14, pady=12)
        inner.pack(fill="both", expand=True)

        # ── título + fechar ───────────────────────────────────────────────────
        top = tk.Frame(inner, bg=BG)
        top.pack(fill="x")

        tk.Label(top, text="👁  Study Tracker", bg=BG, fg=PRIMARY,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(top, text="✕", bg=BG, fg=MUTED, bd=0, cursor="hand2",
                  font=("Segoe UI", 10), activebackground=BG, activeforeground=DANGER,
                  command=self.root.destroy).pack(side="right")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))

        # ── timer ─────────────────────────────────────────────────────────────
        self._timer_var = tk.StringVar(value="00:00")
        self._status_var = tk.StringVar(value="Aguardando...")

        timer_frame = tk.Frame(inner, bg=BG)
        timer_frame.pack(fill="x")
        tk.Label(timer_frame, textvariable=self._timer_var, bg=BG, fg=TEXT,
                 font=("Segoe UI", 26, "bold")).pack(side="left")
        tk.Label(timer_frame, textvariable=self._status_var, bg=BG, fg=SUCCESS,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(10, 0), pady=(8, 0))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(8, 8))

        # ── métricas 2x2 ─────────────────────────────────────────────────────
        grid = tk.Frame(inner, bg=BG)
        grid.pack(fill="x")

        self._vars = {
            "focus":      tk.StringVar(value="—"),
            "dist":       tk.StringVar(value="0"),
            "side":       tk.StringVar(value="0"),
            "lost":       tk.StringVar(value="0"),
        }

        items = [
            ("Foco",           self._vars["focus"], PRIMARY),
            ("Distrações",     self._vars["dist"],  DANGER),
            ("Ol. evasivos",   self._vars["side"],  WARNING),
            ("Perda de foco",  self._vars["lost"],  YELLOW),
        ]

        for i, (label, var, color) in enumerate(items):
            col = i % 2
            row = i // 2
            cell = tk.Frame(grid, bg=BG_CARD, padx=8, pady=6,
                            highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            tk.Label(cell, textvariable=var, bg=BG_CARD, fg=color,
                     font=("Segoe UI", 16, "bold")).pack()
            tk.Label(cell, text=label, bg=BG_CARD, fg=MUTED,
                     font=("Segoe UI", 8)).pack()

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(10, 8))

        # ── botão iniciar ─────────────────────────────────────────────────────
        self._btn_var = tk.StringVar(value="▶  Iniciar Sessão")
        self._action_btn = tk.Button(
            inner, textvariable=self._btn_var, bg=PRIMARY, fg="#fff",
            bd=0, cursor="hand2", font=("Segoe UI", 10, "bold"),
            padx=12, pady=6, activebackground="#1677cc", activeforeground="#fff",
            command=self._toggle_session
        )
        self._action_btn.pack(fill="x")

        # ── botão verificar rastreamento ──────────────────────────────────────
        tk.Button(
            inner, text="Verificar rastreamento", bg=BG_CARD, fg=MUTED,
            bd=0, cursor="hand2", font=("Segoe UI", 8),
            padx=8, pady=4, activebackground=BORDER,
            command=self._open_check
        ).pack(fill="x", pady=(5, 0))

        self._schedule_tick()

    # ── Layout de resumo ──────────────────────────────────────────────────────

    def _build_summary(self, stats):
        self._clear()
        self._mode = "summary"

        root = self.root
        root.geometry("340x440")
        self._center_top_right()

        outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        inner = tk.Frame(outer, bg=BG, padx=16, pady=14)
        inner.pack(fill="both", expand=True)

        # título
        top = tk.Frame(inner, bg=BG)
        top.pack(fill="x")
        tk.Label(top, text="👁  Resumo da Sessão", bg=BG, fg=PRIMARY,
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Button(top, text="✕", bg=BG, fg=MUTED, bd=0, cursor="hand2",
                  font=("Segoe UI", 10), activebackground=BG, activeforeground=DANGER,
                  command=self.root.destroy).pack(side="right")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(10, 12))

        # duração
        dur = stats.get("duration_secs", 0)
        mins, secs = int(dur // 60), int(dur % 60)
        focus_pct = stats.get("focus_percentage", 0)
        focus_color = SUCCESS if focus_pct >= 70 else (WARNING if focus_pct >= 40 else DANGER)

        dur_frame = tk.Frame(inner, bg=BG)
        dur_frame.pack(fill="x")
        tk.Label(dur_frame, text=f"{mins}m {secs}s", bg=BG, fg=TEXT,
                 font=("Segoe UI", 28, "bold")).pack(side="left")
        tk.Label(dur_frame, text="duração", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 0), pady=(14, 0))

        # foco em destaque
        foco_card = tk.Frame(inner, bg=BG_CARD, padx=12, pady=10,
                             highlightthickness=1, highlightbackground=focus_color)
        foco_card.pack(fill="x", pady=(8, 4))
        tk.Label(foco_card, text=f"{focus_pct:.1f}%", bg=BG_CARD, fg=focus_color,
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(foco_card, text="de foco geral", bg=BG_CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 0))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(10, 6))

        # métricas
        rows = [
            ("Distrações totais",   stats.get("total_distractions", 0),    DANGER),
            ("Olhares evasivos",    stats.get("gaze_away_count", 0),         WARNING),
            ("Perdas de foco",      stats.get("focus_lost_count", 0),       YELLOW),
            ("Tempo distraído",     f"{stats.get('total_distraction_secs', 0):.0f}s", MUTED),
        ]

        for label, value, color in rows:
            row = tk.Frame(inner, bg=BG)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=str(value), bg=BG, fg=color,
                     font=("Segoe UI", 13, "bold"), width=6, anchor="w").pack(side="left")
            tk.Label(row, text=label, bg=BG, fg=TEXT,
                     font=("Segoe UI", 10)).pack(side="left")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(10, 10))

        # botões de exportar
        btn_frame = tk.Frame(inner, bg=BG)
        btn_frame.pack(fill="x")

        tk.Button(btn_frame, text="Baixar PDF", bg=PRIMARY, fg="#fff", bd=0,
                  cursor="hand2", font=("Segoe UI", 9, "bold"),
                  padx=10, pady=6, activebackground="#1677cc",
                  command=lambda: webbrowser.open("http://127.0.0.1:5000/api/export/pdf")
                  ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        tk.Button(btn_frame, text="Baixar CSV", bg=BG_CARD, fg=TEXT, bd=0,
                  cursor="hand2", font=("Segoe UI", 9, "bold"),
                  padx=10, pady=6, activebackground=BORDER,
                  highlightthickness=1, highlightbackground=BORDER,
                  command=lambda: webbrowser.open("http://127.0.0.1:5000/api/export/csv")
                  ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        # nova sessão
        tk.Button(inner, text="Nova Sessão", bg=BG_CARD, fg=MUTED, bd=0,
                  cursor="hand2", font=("Segoe UI", 9),
                  padx=10, pady=5, activebackground=BORDER,
                  command=self._new_session
                  ).pack(fill="x", pady=(8, 0))

    # ── Ações ─────────────────────────────────────────────────────────────────

    def _toggle_session(self):
        if not self.tracker._running:
            self.tracker.start_session()
            self._btn_var.set("⏹  Parar Sessão")
            self._action_btn.configure(bg=DANGER, activebackground="#cc2244")
            self._mode = "running"
        else:
            self.tracker.stop_session()
            stats = self.tracker.get_stats()
            # abre o dashboard no browser com os resultados
            webbrowser.open(f"http://127.0.0.1:{self.port}/results")
            self.root.after(0, lambda: self._build_summary(stats))

    def _new_session(self):
        self._build_compact()
        self._center_top_right()

    def _open_check(self):
        """Abre a tela de verificação do rastreamento em thread separada."""
        import threading
        from app.calibration_check import run_calibration_check
        def run():
            run_calibration_check(self.tracker.estimator, self.tracker.camera_index)
        threading.Thread(target=run, daemon=True).start()

    # ── Tick de atualização ───────────────────────────────────────────────────

    def _schedule_tick(self):
        self._tick()

    def _tick(self):
        if self._mode in ("idle", "running") and self.root.winfo_exists():
            self._update_compact()
            self._timer_job = self.root.after(500, self._tick)

    def _update_compact(self):
        session = self.tracker.session
        running = self.tracker._running

        if session and running:
            elapsed = time.time() - session.start_time
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            self._timer_var.set(f"{m:02d}:{s:02d}")

            raw = self.tracker.last_raw
            distracted = raw.get("distracted", False)
            self._status_var.set("DISTRAÍDO" if distracted else "FOCADO")

            # cor do status
            status_widget = None
            for w in self.root.winfo_children():
                status_widget = self._find_label(w, self._status_var)
                if status_widget:
                    break
            if status_widget:
                status_widget.configure(fg=DANGER if distracted else SUCCESS)

            stats = session.to_dict()
            self._vars["focus"].set(f"{stats['focus_percentage']:.0f}%")
            self._vars["dist"].set(str(stats["total_distractions"]))
            self._vars["side"].set(str(stats["gaze_away_count"]))
            self._vars["lost"].set(str(stats["focus_lost_count"]))

    def _find_label(self, widget, var):
        if isinstance(widget, tk.Label):
            try:
                if widget.cget("textvariable") == str(var):
                    return widget
            except Exception:
                pass
        for child in widget.winfo_children():
            found = self._find_label(child, var)
            if found:
                return found
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear(self):
        if self._timer_job:
            try:
                self.root.after_cancel(self._timer_job)
            except Exception:
                pass
            self._timer_job = None
        for w in self.root.winfo_children():
            w.destroy()

    def _center_top_right(self):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self.root.geometry(f"+{sw - w - 20}+{sh - h - 60}")

    def _bind_drag(self):
        self.root.bind("<ButtonPress-1>",   self._drag_start)
        self.root.bind("<B1-Motion>",       self._drag_move)

    def _drag_start(self, e):
        self._drag_x = e.x
        self._drag_y = e.y

    def _drag_move(self, e):
        x = self.root.winfo_x() + e.x - self._drag_x
        y = self.root.winfo_y() + e.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")
