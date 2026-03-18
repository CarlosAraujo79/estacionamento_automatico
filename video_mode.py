from __future__ import annotations

import os
import tempfile
from typing import Optional

import streamlit as st

from plate_ocr import (
    detect_best_plate,
    ocr_gemini,
    ocr_tesseract,
    preprocess_variants_for_ocr,
    _pad_xyxy,
    _import_cv2,
)
from utils import best_whitelist_match, interpret_plate
from whitelist import decide_gate
from parking import occupy_random_spot, render_parking_grid, save_parking_spots


def run_video_mode(
    *,
    model_path: str,
    conf: float,
    device: str,
    pad: float,
    frame_stride: int,
    max_frames: int,
    save_annotated: bool,
    tess_lang: str,
    tess_psm: int,
    gemini_model: str,
    gemini_api_key: str,
    simulate_gate: bool,
    allowed_set: set[str],
    parking_spots: list[Optional[str]],
    to_rgb,
) -> list[Optional[str]]:
    upv = st.file_uploader("Envie um vídeo (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv", "webm"])
    if not upv:
        st.info("Envie um vídeo para começar.")
        return parking_spots

    cv2 = _import_cv2()
    import numpy as np

    suffix = os.path.splitext(upv.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upv.getvalue())
        video_path = f.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Não consegui abrir o vídeo (VideoCapture falhou).")
        return parking_spots

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    st.caption(f"fps={fps:.2f} resolução={w}x{h} frames={total or 'desconhecido'}")

    out_path = None
    writer = None
    if save_annotated:
        out_path = os.path.join(tempfile.gettempdir(), "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

    preview_slot = st.empty()
    prog = st.progress(0)

    frame_idx = 0
    processed = 0
    best_crop = None
    best_conf = -1.0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if max_frames and processed >= int(max_frames):
                break

            if frame_idx % int(frame_stride) != 0:
                if writer is not None:
                    writer.write(frame_bgr)
                frame_idx += 1
                continue

            processed += 1
            frame_idx += 1

            try:
                det = detect_best_plate(model_path, frame_bgr, conf=conf, device=device)
                x1, y1, x2, y2 = _pad_xyxy(*det.xyxy, pad_ratio=pad, w=frame_bgr.shape[1], h=frame_bgr.shape[0])
                if float(det.conf) > best_conf:
                    best_conf = float(det.conf)
                    best_crop = frame_bgr[y1:y2, x1:x2].copy()
                boxed = frame_bgr.copy()
                cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    boxed,
                    f"plate conf={det.conf:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                boxed = frame_bgr

            preview_slot.image(to_rgb(boxed), caption=f"frame={frame_idx}", use_column_width=True)
            if writer is not None:
                writer.write(boxed)
            if total:
                prog.progress(min(1.0, frame_idx / total))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            os.unlink(video_path)
        except Exception:
            pass

    prog.progress(1.0)
    st.success("Processamento finalizado.")

    if out_path and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            st.download_button(
                "Baixar vídeo anotado (mp4)",
                data=f,
                file_name="annotated.mp4",
                mime="video/mp4",
            )

    if simulate_gate:
        st.divider()
        st.subheader("Simulação de estacionamento (vídeo)")
        if best_crop is None or best_crop.size == 0:
            st.error("BLOQUEADO ❌ nenhuma placa detectada no vídeo.")
            render_parking_grid(parking_spots)
            return parking_spots

        st.caption(f"Usando o melhor crop (conf={best_conf:.2f}) para OCR.")
        st.image(to_rgb(best_crop), caption="Melhor crop encontrado", use_column_width=True)

        tess_text_v: Optional[str] = None
        gem_text_v: Optional[str] = None
        colv1, colv2 = st.columns(2)

        with colv1:
            st.markdown("**Tesseract (vídeo)**")
            try:
                variants = preprocess_variants_for_ocr(best_crop)
                best_name = ""
                best_final = ""
                best_raw = ""

                for name, img in variants.items():
                    raw = ocr_tesseract(img, psm=int(tess_psm), lang=tess_lang)
                    final = best_whitelist_match(raw, allowed_set) or interpret_plate(raw)
                    if final:
                        best_name, best_final, best_raw = name, final, raw
                        break
                    if not best_raw and raw:
                        best_name, best_raw = name, raw

                tess_text_v = best_raw
                st.code(best_final or (tess_text_v or ""), language=None)
            except Exception as e:
                st.warning(f"Tesseract falhou: {e}")

        with colv2:
            st.markdown("**Gemini (vídeo)**")
            if not gemini_api_key:
                st.warning("Gemini: sem API key configurada.")
            else:
                try:
                    gem_text_v = ocr_gemini(best_crop, api_key=gemini_api_key, model_name=gemini_model)
                    st.code(best_whitelist_match(gem_text_v, allowed_set) or interpret_plate(gem_text_v), language=None)
                except Exception as e:
                    st.warning(f"Gemini falhou: {e}")

        summary_v = {
            "tesseract": best_whitelist_match(tess_text_v, allowed_set) or interpret_plate(tess_text_v),
            "gemini": best_whitelist_match(gem_text_v, allowed_set) or interpret_plate(gem_text_v),
        }
        st.write(summary_v)
        allowed, matched = decide_gate(allowed_set, [summary_v["gemini"], summary_v["tesseract"]])
        if allowed:
            st.success(f"LIBERADO ✅ placa={matched}")
            parking_spots, filled_idx = occupy_random_spot(parking_spots, matched)
            save_parking_spots(parking_spots)
            if filled_idx is None:
                st.warning("Estacionamento lotado (30/30).")
            else:
                st.info(f"Vaga preenchida: V{filled_idx+1:02d}")
        else:
            st.error("BLOQUEADO ❌ placa não autorizada (ou OCR vazio)")

        st.divider()
        render_parking_grid(parking_spots)

    return parking_spots

