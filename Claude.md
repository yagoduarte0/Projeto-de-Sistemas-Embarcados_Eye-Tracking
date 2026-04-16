# Eye Tracking Study Dashboard

Projeto acadêmico (5° período) — dashboard de monitoramento de atenção durante estudos via webcam.

## Stack
- **Eye tracking:** [EyeTrax](https://github.com/ck-zhang/EyeTrax) (MediaPipe)
- **Backend:** Flask + Flask-SocketIO
- **Frontend:** HTML/CSS/JS + Chart.js
- **Relatórios:** fpdf2 (PDF) + csv (CSV)

## Estrutura
```
eye-tracking-studying/
├── main.py              # entrada: calibração + servidor
├── app/
│   ├── tracker.py       # wrapper EyeTrax + detecção de distração
│   ├── server.py        # Flask + SocketIO + rotas de API
│   └── reports.py       # exportação PDF e CSV
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
├── templates/index.html
└── requirements.txt
```

## Como rodar
```bash
# Ativar venv
source venv/Scripts/activate   # Windows

# Primeira vez (com calibração)
python main.py

# Pular calibração (usa modelo salvo)
python main.py --skip-cal
```

## O que é detectado
| Evento | Como | Threshold |
|---|---|---|
| Olhar lateral | Razão iris < 0.35 | `SIDE_GAZE_RATIO` |
| Distração (gaze fora da tela) | Coordenada X/Y fora da tela + margem | `SCREEN_MARGIN` |
| Perda de foco / sonolência | Olhos fechados > 2s | `BLINK_SUSTAINED_SECS` |
| Alerta | Distraído por > 4s contínuos | `ALERT_DISTRACTION_SECS` |

## API
- `POST /api/start` — inicia sessão
- `POST /api/stop`  — para sessão
- `GET  /api/stats` — dados da sessão atual
- `GET  /api/export/pdf` — relatório PDF
- `GET  /api/export/csv` — relatório CSV

## Dependências principais
```
eyetrax, flask, flask-socketio, fpdf2, mediapipe, opencv-python
```
