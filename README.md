# 👁️ Eye Tracking — Sistemas Embarcados

Sistema de rastreamento ocular em tempo real desenvolvido como projeto da disciplina de **Sistemas Embarcados**. A aplicação captura o olhar do usuário através da webcam, processa os dados com um modelo de Machine Learning e exibe o resultado em uma interface web interativa.

---

## 📋 Sobre o Projeto

O sistema utiliza visão computacional para detectar os pontos de referência faciais e oculares do usuário, estima a direção do olhar (*gaze estimation*) e mapeia as coordenadas para a tela. A interface é servida via **Flask** com comunicação em tempo real por **WebSockets**.

---

## 🚀 Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Linguagem principal |
| Flask + Flask-SocketIO | Servidor web e comunicação em tempo real |
| OpenCV | Captura e processamento de vídeo |
| MediaPipe | Detecção de landmarks faciais |
| scikit-learn | Modelo de regressão para gaze estimation |
| NumPy / SciPy | Processamento numérico |
| eyetrax | Utilitários de eye tracking |
| fpdf2 | Geração de relatórios em PDF |
| screeninfo | Obtenção de resolução da tela |
| HTML / CSS / JavaScript | Interface web |

---

## 📁 Estrutura do Projeto

```
Projeto-de-Sistemas-Embarcados_Eye-Tracking/
│
├── app/                  # Módulos da aplicação Flask
├── static/               # Arquivos estáticos (CSS, JS, imagens)
├── templates/            # Templates HTML (Jinja2)
├── main.py               # Ponto de entrada da aplicação
├── gaze_model.pkl        # Modelo de ML treinado (scikit-learn)
├── requirements.txt      # Dependências do projeto
├── Claude.md             # Notas de desenvolvimento com IA
└── .gitignore
```

---

## ⚙️ Instalação e Execução

### Pré-requisitos

- Python 3.9 ou superior
- Webcam funcional
- pip atualizado

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/yagoduarte0/Projeto-de-Sistemas-Embarcados_Eye-Tracking.git
cd Projeto-de-Sistemas-Embarcados_Eye-Tracking

# 2. Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
python main.py
```

Acesse em: **http://localhost:5000**

---

## 📦 Dependências

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

## 🧠 Como Funciona

1. A webcam captura frames em tempo real.
2. O **MediaPipe** detecta os landmarks do rosto e dos olhos.
3. As features extraídas são enviadas ao modelo `gaze_model.pkl` (treinado com **scikit-learn**) para estimar a direção do olhar.
4. As coordenadas do olhar são transmitidas ao frontend via **WebSocket** (Flask-SocketIO).
5. A interface web exibe o ponto de gaze sobre a tela em tempo real.

---

## 📄 Licença

Projeto acadêmico desenvolvido para fins educacionais no IBMEC — Barra da Tijuca, Rio de Janeiro.

---

## 👤 Autor

**Yago Duarte**  
**Fabrício de Brito**
**Paco Guimarães**
