import os
import tempfile
import warnings
from typing import Optional

import streamlit as st

from plate_ocr import Detection, _import_cv2, _pad_xyxy
from utils import norm_plate
from whitelist import load_allowed_plates, save_allowed_plates
from parking import (
    free_random_occupied,
    load_parking_spots,
    occupy_random_spot,
    render_parking_grid,
    save_parking_spots,
)
from image_mode import run_image_mode
from camera_mode import run_camera_mode
from live_camera_mode import run_live_camera_mode
from video_mode import run_video_mode

# Evita ruído no console em ambientes com múltiplas instalações do Matplotlib.
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)


def _bgr_from_uploaded_file(data: bytes):
    cv2 = _import_cv2()
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não consegui decodificar a imagem enviada.")
    return img


def _to_rgb(bgr):
    cv2 = _import_cv2()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crop_from_det(img_bgr, det: Detection, pad: float):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _pad_xyxy(*det.xyxy, pad_ratio=pad, w=w, h=h)
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


@st.cache_resource
def _load_model_path(path: str) -> str:
    # ultralytics YOLO faz cache interno; aqui só cacheamos a string para evitar reload acidental do app
    return path


def main():
    st.set_page_config(page_title="Detecção de Placas + OCR", layout="wide")
    st.title("Detecção de Placas + OCR (Tesseract + Gemini)")

    with st.sidebar:
        st.header("Configurações")
        page = st.radio("Página", options=["Detecção", "Estacionamento"])
        simulate_gate = st.toggle("Simular estacionamento (liberar por whitelist)", value=True)
        mode = st.radio(
            "Entrada",
            options=["Imagem (upload)", "Câmera do dispositivo", "Câmera ao vivo (auto)", "Vídeo (upload)"],
            disabled=(page != "Detecção"),
        )
        model_path = st.text_input("Modelo (.pt)", value="plaquinhas.pt")
        conf = st.slider("Confidence mínimo (detecção)", 0.01, 0.90, 0.25, 0.01)
        device = st.text_input("Device (cpu, 0, 0,1 ...)", value="cpu")
        pad = st.slider("Padding no crop", 0.00, 0.40, 0.08, 0.01)

        st.divider()
        st.subheader("Placas autorizadas (whitelist)")
        allowed_list = load_allowed_plates()
        allowed_text = st.text_area(
            "Uma por linha",
            value="\n".join(allowed_list),
            height=140,
            help="As placas são normalizadas (maiúsculas, sem espaço/hífen).",
        )
        if st.button("Salvar whitelist"):
            plates = [norm_plate(x) for x in allowed_text.splitlines()]
            save_allowed_plates([p for p in plates if p])
            st.success("Whitelist salva.")
            allowed_list = load_allowed_plates()

        if mode == "Câmera ao vivo (auto)":
            st.divider()
            st.subheader("Leitura automática")
            interval_s = st.slider("Intervalo entre leituras (segundos)", 1, 10, 3, 1)
            mirror_x = st.checkbox("Espelhar vídeo (flip no eixo X)", value=True)
        else:
            interval_s = 3
            mirror_x = True

        if mode == "Vídeo (upload)":
            st.divider()
            st.subheader("Vídeo")
            frame_stride = st.slider("Processar a cada N frames", 1, 10, 2, 1)
            max_frames = st.number_input("Máximo de frames (0 = sem limite)", min_value=0, value=0, step=50)
            save_annotated = st.checkbox("Gerar vídeo anotado (bbox)", value=True)
        else:
            frame_stride = 1
            max_frames = 0
            save_annotated = False

        st.divider()
        st.subheader("Tesseract")
        tess_lang = st.text_input("Idioma (lang)", value="eng")
        tess_psm = st.selectbox("PSM", options=[6, 7, 8, 11, 13], index=1)

        st.divider()
        st.subheader("Gemini")
        gemini_model = st.text_input("Modelo", value="gemini-2.5-flash")
        gemini_api_key = st.text_input("API Key (opcional)", value="", type="password")

    # Preferir secrets/env para não exigir digitar na UI.
    secrets_key = ""
    try:
        secrets_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        secrets_key = ""
    effective_gemini_key = secrets_key or os.getenv("GEMINI_API_KEY", "").strip() or gemini_api_key.strip()

    allowed_set = set(load_allowed_plates())
    if "parking_spots" not in st.session_state:
        st.session_state["parking_spots"] = load_parking_spots()
    parking_spots: list[Optional[str]] = st.session_state["parking_spots"]

    if page == "Estacionamento":
        st.title("Estacionamento (visualização)")
        render_parking_grid(parking_spots)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Resetar estacionamento (esvaziar tudo)"):
                parking_spots = [None] * 30
                save_parking_spots(parking_spots)
                st.session_state["parking_spots"] = parking_spots
                st.success("Estacionamento resetado.")
        with c2:
            if st.button("Liberar vaga aleatória (saída)"):
                parking_spots, freed = free_random_occupied(parking_spots)
                save_parking_spots(parking_spots)
                st.session_state["parking_spots"] = parking_spots
                if freed is None:
                    st.warning("Não há vagas ocupadas para liberar.")
                else:
                    st.info(f"Vaga liberada: V{freed+1:02d}")
        with c3:
            plate_in = st.text_input("Simular entrada manual (placa)", value="")
            if st.button("Preencher vaga aleatória (manual)"):
                plate = norm_plate(plate_in)
                if not plate:
                    st.warning("Informe uma placa.")
                else:
                    parking_spots, filled = occupy_random_spot(parking_spots, plate)
                    save_parking_spots(parking_spots)
                    st.session_state["parking_spots"] = parking_spots
                    if filled is None:
                        st.warning("Estacionamento lotado (30/30).")
                    else:
                        st.info(f"Vaga preenchida: V{filled+1:02d}")
        return

    # Página Detecção
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado: {model_path}")
        return

    if mode == "Imagem (upload)":
        parking_spots = run_image_mode(
            model_path=model_path,
            conf=conf,
            device=device,
            pad=pad,
            tess_lang=tess_lang,
            tess_psm=int(tess_psm),
            gemini_model=gemini_model,
            gemini_api_key=effective_gemini_key,
            simulate_gate=simulate_gate,
            allowed_set=allowed_set,
            parking_spots=parking_spots,
            bgr_from_uploaded=_bgr_from_uploaded_file,
            to_rgb=_to_rgb,
            crop_from_det=_crop_from_det,
        )
        st.session_state["parking_spots"] = parking_spots
    elif mode == "Câmera do dispositivo":
        parking_spots = run_camera_mode(
            model_path=model_path,
            conf=conf,
            device=device,
            pad=pad,
            tess_lang=tess_lang,
            tess_psm=int(tess_psm),
            gemini_model=gemini_model,
            gemini_api_key=effective_gemini_key,
            simulate_gate=simulate_gate,
            allowed_set=allowed_set,
            parking_spots=parking_spots,
            bgr_from_uploaded=_bgr_from_uploaded_file,
            to_rgb=_to_rgb,
            crop_from_det=_crop_from_det,
        )
        st.session_state["parking_spots"] = parking_spots
    elif mode == "Câmera ao vivo (auto)":
        run_live_camera_mode(
            model_path=model_path,
            conf=conf,
            device=device,
            pad=pad,
            interval_s=float(interval_s),
            mirror_x=bool(mirror_x),
            tess_lang=tess_lang,
            tess_psm=int(tess_psm),
            gemini_model=gemini_model,
            gemini_api_key=effective_gemini_key,
            simulate_gate=simulate_gate,
            allowed_set=allowed_set,
        )
    else:
        parking_spots = run_video_mode(
            model_path=model_path,
            conf=conf,
            device=device,
            pad=pad,
            frame_stride=int(frame_stride),
            max_frames=int(max_frames),
            save_annotated=bool(save_annotated),
            tess_lang=tess_lang,
            tess_psm=int(tess_psm),
            gemini_model=gemini_model,
            gemini_api_key=effective_gemini_key,
            simulate_gate=simulate_gate,
            allowed_set=allowed_set,
            parking_spots=parking_spots,
            to_rgb=_to_rgb,
        )
        st.session_state["parking_spots"] = parking_spots


if __name__ == "__main__":
    main()

