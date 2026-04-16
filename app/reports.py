"""
Geração de relatórios PDF e CSV a partir dos dados de sessão.
"""
import csv
import io
import time
from fpdf import FPDF


def export_csv(stats: dict) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Relatório de Sessão de Estudos"])
    writer.writerow(["Gerado em", time.strftime("%d/%m/%Y %H:%M")])
    writer.writerow([])

    writer.writerow(["Métrica", "Valor"])
    writer.writerow(["Duração (s)", stats.get("duration_secs", 0)])
    writer.writerow(["Foco (%)", stats.get("focus_percentage", 0)])
    writer.writerow(["Total de distrações", stats.get("total_distractions", 0)])
    writer.writerow(["Olhares laterais", stats.get("side_gaze_count", 0)])
    writer.writerow(["Perdas de foco", stats.get("focus_lost_count", 0)])
    writer.writerow(["Tempo distraído (s)", stats.get("total_distraction_secs", 0)])
    writer.writerow([])

    writer.writerow(["Linha do tempo de eventos"])
    writer.writerow(["Tipo", "Tempo na sessão (s)", "Detalhe"])
    for ev in stats.get("events", []):
        writer.writerow([ev["kind"], ev["timestamp"], ev["detail"]])

    return output.getvalue().encode("utf-8-sig")


class StudyPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_fill_color(30, 144, 255)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "  Relatório de Sessão de Estudos", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Pagina {self.page_no()} - Eye Tracking Study Dashboard", align="C")


def export_pdf(stats: dict) -> bytes:
    pdf = StudyPDF()
    pdf.add_page()

    generated_at = time.strftime("%d/%m/%Y às %H:%M")
    duration = stats.get("duration_secs", 0)
    mins = int(duration // 60)
    secs = int(duration % 60)

    # ── Cabeçalho de data ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100)
    pdf.cell(0, 6, f"Gerado em {generated_at}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Cartões de métricas ──────────────────────────────────────────────────
    metrics = [
        ("Duração da sessão", f"{mins}m {secs}s"),
        ("Foco geral", f"{stats.get('focus_percentage', 0):.1f}%"),
        ("Distrações totais", str(stats.get("total_distractions", 0))),
        ("Olhares para o lado", str(stats.get("side_gaze_count", 0))),
        ("Perdas de foco", str(stats.get("focus_lost_count", 0))),
        ("Tempo distraído", f"{stats.get('total_distraction_secs', 0):.0f}s"),
    ]

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0)

    card_w = (pdf.w - pdf.l_margin - pdf.r_margin - 8) / 2
    card_h = 20
    col = 0
    x_start = pdf.l_margin

    for label, value in metrics:
        x = x_start + col * (card_w + 8)
        y = pdf.get_y()

        pdf.set_fill_color(245, 247, 250)
        pdf.set_draw_color(200, 210, 220)
        pdf.rect(x, y, card_w, card_h, style="FD")

        pdf.set_xy(x + 3, y + 2)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100)
        pdf.cell(card_w - 6, 5, label)

        pdf.set_xy(x + 3, y + 8)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 144, 255)
        pdf.cell(card_w - 6, 10, value)

        col += 1
        if col == 2:
            col = 0
            pdf.ln(card_h + 4)

    if col != 0:
        pdf.ln(card_h + 4)

    pdf.ln(6)

    # ── Linha do tempo ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0)
    pdf.cell(0, 8, "Linha do Tempo de Eventos", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    events = stats.get("events", [])
    if not events:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(150)
        pdf.cell(0, 8, "Nenhum evento registrado.", new_x="LMARGIN", new_y="NEXT")
    else:
        # cabeçalho da tabela
        pdf.set_fill_color(30, 144, 255)
        pdf.set_text_color(255)
        pdf.set_font("Helvetica", "B", 10)
        col_widths = [35, 40, 100]
        headers = ["Tempo (s)", "Tipo", "Detalhe"]
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 8, h, border=1, fill=True)
        pdf.ln()

        kind_labels = {
            "side_gaze": "Olhar lateral",
            "distraction": "Distração",
            "focus_lost": "Perda de foco",
            "refocus": "Refoco",
        }

        for i, ev in enumerate(events):
            fill = i % 2 == 0
            pdf.set_fill_color(245, 247, 250) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(0)
            pdf.set_font("Helvetica", "", 9)

            row = [
                f"{ev['timestamp']:.1f}",
                kind_labels.get(ev["kind"], ev["kind"]),
                ev.get("detail", ""),
            ]
            for i, val in enumerate(row):
                pdf.cell(col_widths[i], 7, val, border=1, fill=fill)
            pdf.ln()

    return bytes(pdf.output())
