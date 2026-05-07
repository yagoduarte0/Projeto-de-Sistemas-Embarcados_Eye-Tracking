# Eye Tracking Study Dashboard

Sistema de monitoramento de atenção em tempo real via webcam, desenvolvido como projeto da disciplina de **Sistemas Embarcados** (5º período — IBMEC, Barra da Tijuca, RJ).

O sistema aplica um pipeline de processamento de sinais biomecânicos — filtro de Kalman, fusão multimodal e calibração adaptativa — para produzir o **IAF (Índice de Atenção Focalizada)**, um score contínuo [0, 1] que representa o nível de atenção do usuário frame a frame, sem hardware especializado.

---

## Tecnologias

| Tecnologia | Uso |
|---|---|
| Python 3.10 | Linguagem principal |
| MediaPipe FaceLandmarker | 478 landmarks faciais por frame |
| OpenCV | Captura de vídeo (câmera persistente) |
| Flask + Flask-SocketIO | Servidor web + WebSocket em tempo real |
| fpdf2 | Relatórios PDF |
| Tkinter | Overlay always-on-top |
| HTML / CSS / Chart.js | Dashboard web |

---

## Pipeline de Sinais

```
Webcam → MediaPipe (478 landmarks) → Filtro de Kalman 1D
       → Calibração 5 pontos → IAF (fusão ponderada)
       → Detecção de distração → WebSocket → Dashboard
```

**Sinais extraídos por frame:**
- `iris_h` — posição horizontal da íris (razão entre cantos do olho)
- `iris_v` — posição vertical da íris (razão entre pálpebras)
- `head_yaw` — rotação horizontal da cabeça (desvio do nariz)
- `avg_ear` — Eye Aspect Ratio (abertura dos olhos)

**Filtro de Kalman 1D** (q = 1×10⁻⁵, r = 5×10⁻³): suaviza os sinais de íris, separando movimento real de ruído dos landmarks. Dois limiares de EAR independentes: `0.15` para congelar o filtro (íris invisível) e `0.22` para o guard de detecção.

**IAF — Índice de Atenção Focalizada:**

```
IAF = 0.45 × iris_h  +  0.20 × iris_v  +  0.20 × yaw_score  +  0.15 × ear_smooth
```

| Componente | Peso | Descrição |
|---|---|---|
| iris_h | 0.45 | Desvio horizontal da íris (mais discriminativo) |
| iris_v | 0.20 | Desvio vertical da íris |
| yaw_score | 0.20 | Alinhamento da cabeça |
| ear_smooth | 0.15 | Abertura ocular (anti-sonolência, imune a piscadas) |

---

## Calibração Adaptativa

A calibração de **5 pontos** (centro + 4 cantos da tela) mapeia os limites reais do campo visual de cada usuário em cada dispositivo. Necessária uma vez por setup (monitor + posição da câmera).

- **Tempo total:** 12,5 s (2,5 s por ponto, 0,7 s de estabilização + 1,8 s de coleta)
- **Saída:** limites de detecção assimétricos por borda + zona neutra do IAF
- **Persistência:** `~/.cache/eyetrax/study_tracker_calib.json`

Sem calibração, thresholds fixos são usados como fallback.

---

## Detecção de Eventos

| Evento | Condição | Parâmetro |
|---|---|---|
| `side_gaze` | Íris fora dos limites calibrados por ≥ 5 frames | `SMOOTH_FRAMES = 5` |
| `focus_lost` | Olhos fechados por > 2,5 s contínuos | `BLINK_SUSTAINED_SECS` |
| `focus_lost` | Distração contínua por > 5 s | `ALERT_DISTRACTION_SECS` |

Latência de detecção: **~167 ms** @ 30 fps. Latência de recuperação: **~100 ms** (máximo).

**Compensação VOR:** quando a cabeça gira, a íris compensa naturalmente na direção oposta (reflexo vestíbulo-ocular). O sistema detecta esse padrão e suprime falsos positivos, exigindo giro de cabeça ≥ 0,09 para ativar a compensação.

---

## Estrutura do Projeto

```
├── main.py                  # entrada — calibração + servidor + overlay
├── app/
│   ├── tracker.py           # pipeline de sinais, Kalman, IAF, detecção
│   ├── server.py            # Flask + SocketIO + 12 endpoints REST
│   ├── overlay.py           # overlay tkinter always-on-top
│   ├── calibration_check.py # calibração de 5 pontos (tkinter fullscreen)
│   └── reports.py           # exportação PDF e CSV
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
├── templates/
│   ├── index.html           # dashboard ao vivo
│   └── results.html         # resumo pós-sessão
└── requirements.txt
```

---

## Como Rodar

```bash
# 1. Clone o repositório
git clone https://github.com/yagoduarte0/Projeto-de-Sistemas-Embarcados_Eye-Tracking.git
cd Projeto-de-Sistemas-Embarcados_Eye-Tracking

# 2. Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/macOS

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute
python main.py                # abre calibração na 1ª vez, depois inicia
python main.py --calibrate    # força recalibração
python main.py --port 8080    # porta alternativa
```

Acesse em **http://127.0.0.1:5000**

Na primeira execução, uma tela fullscreen de calibração de 5 pontos abrirá automaticamente. O overlay tkinter aparece no canto superior direito da tela.

---

## API

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/start` | Inicia sessão |
| POST | `/api/stop` | Para sessão |
| POST | `/api/resume` | Retoma sessão pausada |
| POST | `/api/calibrate` | Dispara calibração via web |
| GET | `/api/stats` | Estatísticas da sessão atual |
| GET | `/api/gaze_history` | Histórico de posição da íris |
| GET | `/api/export/pdf` | Relatório PDF |
| GET | `/api/export/csv` | Dados CSV (por sub-sessão) |

**Eventos WebSocket** emitidos em tempo real:

| Evento | Frequência | Conteúdo |
|---|---|---|
| `frame` | ~30 Hz | JPEG + IAF + FPS + latência + distracted |
| `stats_update` | por evento | Estatísticas agregadas da sessão |
| `new_event` | por distração | Tipo, timestamp, sub-sessão |
| `alert` | a cada 5 s | Alerta de distração prolongada |

---

## Sub-sessões

É possível pausar e retomar uma sessão sem perder os dados acumulados. O timer continua de onde parou e cada trecho é identificado como sub-sessão no relatório:

```
=== Sessão 1 ===
side_gaze, 30.1s, olhando para esquerda

=== Sessão 2 ===
side_gaze, 185.2s, olhando para direita
```

---

## Dependências

```
eyetrax>=0.4.0
flask>=3.0
flask-socketio>=5.0
fpdf2>=2.8
numpy<2
opencv-python>=4.5
mediapipe>=0.10
scikit-learn>=1.3
scipy>=1.10
screeninfo>=0.8
```

---

## Autores

**Yago Duarte** · **Fabrício de Brito** · **Paco Guimarães**  
IBMEC — Barra da Tijuca, Rio de Janeiro · Projeto Acadêmico 2025
