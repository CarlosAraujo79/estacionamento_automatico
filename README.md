# Placas (detecção + OCR)

Este projeto usa seu modelo `.pt` (YOLO via `ultralytics`) para **detectar a placa na imagem**, recortar (crop) a região e depois fazer OCR via:

- **Tesseract** (offline, rápido, sem custos)
- **Gemini 2.5 Flash** (online, costuma ler melhor em condições ruins)

## Requisitos

### Python deps

Para o Streamlit Cloud, use `opencv-python-headless` (já está em `requirements.txt`).

No Ubuntu/Debian, instale o suporte a venv:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tesseract (para OCR offline)

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## Uso

## Interface (recomendado)

Rode a interface:

```bash
streamlit run app.py
```

### Configurar a API key (sem colocar na tela)

Opção A (recomendado no Streamlit): crie `.streamlit/secrets.toml` a partir do exemplo:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edite `.streamlit/secrets.toml` e coloque sua chave em `GEMINI_API_KEY`.

Opção B: variável de ambiente:

```bash
export GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

A interface faz:
- detecção da placa (bbox + crop)
- OCR com **Tesseract e Gemini ao mesmo tempo**
- se um OCR falhar, aparece só um **aviso** e o outro continua
- detecção em **vídeo (upload)** com geração de vídeo anotado para download
- simulação de **estacionamento**: libera/bloqueia se a placa lida estiver na whitelist
- mapa visual de **30 vagas (3×10)** com “luzes” (🟢 livre / 🔴 ocupada) e preenchimento aleatório a cada carro liberado
- detecção por **câmera do dispositivo** (tirar foto na interface)
- detecção por **câmera ao vivo (auto)** com intervalo entre leituras

### 1) Detectar placa + OCR com Tesseract

```bash
python3 plate_ocr.py --image /caminho/para/imagem.jpg --model plaquinhas.pt --ocr tesseract --save-debug
```

Visual (abre janelas com bbox/crop):

```bash
python3 plate_ocr.py --image /caminho/para/imagem.jpg --model plaquinhas.pt --ocr tesseract --show
```

Saídas (se `--save-debug`):
- `outputs/boxed.jpg`: imagem com bbox
- `outputs/crop.jpg`: recorte da placa
- `outputs/crop_preproc.png`: pré-processamento usado no OCR

### 2) Detectar placa + OCR com Gemini 2.5 Flash

Defina sua chave:

```bash
export GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

Rode:

```bash
python3 plate_ocr.py --image /caminho/para/imagem.jpg --model plaquinhas.pt --ocr gemini --save-debug
```

Visual:

```bash
python3 plate_ocr.py --image /caminho/para/imagem.jpg --model plaquinhas.pt --ocr gemini --show
```

## Dicas rápidas

- Se estiver detectando mas cortando “apertado”, aumente `--pad` (ex: `--pad 0.15`).
- Se estiver perdendo detecção, reduza `--conf` (ex: `--conf 0.15`).
- Se tiver GPU, tente `--device 0`.

