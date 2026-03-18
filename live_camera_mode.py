from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from parking import load_parking_spots, occupy_random_spot, render_parking_grid, save_parking_spots
from plate_ocr import (
    _import_cv2,
    ocr_gemini,
    ocr_tesseract,
    preprocess_variants_for_ocr,
)
from utils import best_whitelist_match, interpret_plate
from whitelist import decide_gate


@dataclass
class GateEvent:
    ts: float
    allowed: bool
    plate: str
    source: str
    message: str


def _best_tesseract_from_crop(crop_bgr, allowed_set: set[str], tess_lang: str, tess_psm: int) -> tuple[str, str]:
    """
    Retorna (plate_final, preprocess_name).
    """
    variants = preprocess_variants_for_ocr(crop_bgr)
    for name, img in variants.items():
        raw = ocr_tesseract(img, psm=int(tess_psm), lang=tess_lang)
        final = best_whitelist_match(raw, allowed_set) or interpret_plate(raw)
        if final:
            return final, name
    return "", ""


def run_live_camera_mode(
    *,
    model_path: str,
    conf: float,
    device: str,
    pad: float,
    interval_s: float,
    mirror_x: bool,
    tess_lang: str,
    tess_psm: int,
    gemini_model: str,
    gemini_api_key: str,
    simulate_gate: bool,
    allowed_set: set[str],
) -> None:
    """
    Stream ao vivo da câmera com leitura automática a cada interval_s.
    """
    try:
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except Exception:
        st.error("Faltam deps do modo ao vivo. Instale: `pip install -r requirements.txt`.")
        return

    cv2 = _import_cv2()
    import numpy as np

    try:
        from ultralytics import YOLO
    except Exception as e:
        st.error("Falha ao importar ultralytics. Instale as deps e reinicie.")
        return

    model = YOLO(model_path)

    status = st.empty()
    event_box = st.empty()
    grid_box = st.empty()

    if "parking_spots" not in st.session_state:
        st.session_state["parking_spots"] = load_parking_spots()

    class Processor(VideoProcessorBase):
        def __init__(self):
            self.last_read_ts = 0.0
            self.last_plate = ""
            self.last_msg = "Aguardando leitura..."

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if mirror_x:
                img = cv2.flip(img, 1)

            # Detecção (leve, por frame)
            try:
                results = model.predict(source=img, conf=conf, device=device, verbose=False)
                r0 = results[0] if results else None
                boxes = getattr(r0, "boxes", None) if r0 is not None else None
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confs = boxes.conf.detach().cpu().numpy()
                    best_i = int(np.argmax(confs))
                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy[best_i].tolist()]

                    h, w = img.shape[:2]
                    bw, bh = (x2 - x1), (y2 - y1)
                    xpad = int(round(bw * pad))
                    ypad = int(round(bh * pad))
                    x1 = max(0, min(x1 - xpad, w - 1))
                    y1 = max(0, min(y1 - ypad, h - 1))
                    x2 = max(0, min(x2 + xpad, w - 1))
                    y2 = max(0, min(y2 + ypad, h - 1))
                    if x2 <= x1:
                        x2 = min(w - 1, x1 + 1)
                    if y2 <= y1:
                        y2 = min(h - 1, y1 + 1)

                    # Desenha bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"plate conf={float(confs[best_i]):.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Leitura automática por intervalo
                    now = time.time()
                    if interval_s > 0 and (now - self.last_read_ts) >= interval_s:
                        self.last_read_ts = now
                        crop = img[y1:y2, x1:x2].copy()

                        tess_plate, tess_pp = _best_tesseract_from_crop(crop, allowed_set, tess_lang, tess_psm)
                        gem_plate = ""
                        if gemini_api_key:
                            try:
                                raw = ocr_gemini(crop, api_key=gemini_api_key, model_name=gemini_model)
                                gem_plate = best_whitelist_match(raw, allowed_set) or interpret_plate(raw)
                            except Exception:
                                gem_plate = ""

                        # Decide usando gemini > tesseract
                        final_candidates = [gem_plate, tess_plate]
                        allowed, matched = decide_gate(allowed_set, final_candidates)
                        if simulate_gate and allowed and matched:
                            spots = st.session_state.get("parking_spots") or load_parking_spots()
                            spots, idx = occupy_random_spot(spots, matched)
                            save_parking_spots(spots)
                            st.session_state["parking_spots"] = spots
                            if idx is None:
                                self.last_msg = f"LIBERADO: {matched} | estacionamento lotado"
                            else:
                                self.last_msg = f"LIBERADO: {matched} | vaga V{idx+1:02d}"
                        elif simulate_gate:
                            self.last_msg = f"BLOQUEADO | gemini={gem_plate or '-'} tess={tess_plate or '-'}"
                        else:
                            self.last_msg = f"OCR | gemini={gem_plate or '-'} tess={tess_plate or '-'}"

                        self.last_plate = matched if allowed else (gem_plate or tess_plate)

            except Exception:
                # não derruba o stream
                pass

            return frame.from_ndarray(img, format="bgr24")

    # API nova: video_processor_factory. Mantém fallback para versões antigas.
    try:
        ctx = webrtc_streamer(
            key="live-camera",
            video_processor_factory=Processor,  # type: ignore[call-arg]
            media_stream_constraints={"video": True, "audio": False},
        )
    except TypeError:
        ctx = webrtc_streamer(
            key="live-camera",
            video_transformer_factory=Processor,
            media_stream_constraints={"video": True, "audio": False},
        )

    # Painel de status enquanto está rodando
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc is not None:
            status.info(proc.last_msg)
        spots = st.session_state.get("parking_spots") or load_parking_spots()
        with grid_box.container():
            render_parking_grid(spots)
        time.sleep(0.5)

