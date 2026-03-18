from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from plate_ocr import (
    detect_best_plate,
    draw_detection,
    ocr_gemini,
    ocr_tesseract,
    preprocess_variants_for_ocr,
)
from utils import best_whitelist_match, interpret_plate
from whitelist import decide_gate
from parking import occupy_random_spot, render_parking_grid, save_parking_spots, free_random_occupied


def run_image_mode(
    *,
    model_path: str,
    conf: float,
    device: str,
    pad: float,
    tess_lang: str,
    tess_psm: int,
    gemini_model: str,
    gemini_api_key: str,
    simulate_gate: bool,
    allowed_set: set[str],
    parking_spots: list[Optional[str]],
    bgr_from_uploaded,
    to_rgb,
    crop_from_det,
) -> list[Optional[str]]:
    up = st.file_uploader("Envie uma imagem (jpg/png)", type=["jpg", "jpeg", "png", "webp"])
    if not up:
        st.info("Envie uma imagem para começar.")
        return parking_spots

    try:
        img_bgr = bgr_from_uploaded(up.getvalue())
    except Exception as e:
        st.error(str(e))
        return parking_spots

    colA, colB = st.columns([1.2, 1])
    with st.spinner("Detectando placa..."):
        try:
            det = detect_best_plate(model_path, img_bgr, conf=conf, device=device)
        except Exception as e:
            st.error(f"Falha na detecção: {e}")
            return parking_spots

    crop_bgr, crop_xyxy = crop_from_det(img_bgr, det, pad=pad)
    boxed_bgr = draw_detection(img_bgr, det)

    with colA:
        st.subheader("Detecção")
        st.image(to_rgb(boxed_bgr), caption=f"bbox={crop_xyxy} conf={det.conf:.2f}", use_column_width=True)

    with colB:
        st.subheader("Crop da placa")
        st.image(to_rgb(crop_bgr), use_column_width=True)

    st.divider()
    st.subheader("OCR (rodando os dois em paralelo lógico)")

    ocr_col1, ocr_col2 = st.columns(2)
    tess_text: Optional[str] = None
    gem_text: Optional[str] = None

    with ocr_col1:
        st.markdown("**Tesseract**")
        try:
            variants = preprocess_variants_for_ocr(crop_bgr)
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

            tess_text = best_raw
            st.code(best_final or (tess_text or ""), language=None)
            if best_name:
                st.image(variants[best_name], caption=f"Pré-processamento (Tesseract): {best_name}", use_column_width=True)
        except Exception as e:
            st.warning(f"Tesseract falhou: {e}")

    with ocr_col2:
        st.markdown("**Gemini 2.5 Flash**")
        if not gemini_api_key:
            st.warning("Gemini: sem API key configurada.")
        else:
            try:
                gem_text = ocr_gemini(crop_bgr, api_key=gemini_api_key, model_name=gemini_model)
                st.code(gem_text or "", language=None)
            except Exception as e:
                st.warning(f"Gemini falhou: {e}")

    st.divider()
    st.subheader("Resumo / Simulação de estacionamento")

    tess_final = best_whitelist_match(tess_text, allowed_set) or interpret_plate(tess_text)
    gem_final = best_whitelist_match(gem_text, allowed_set) or interpret_plate(gem_text)
    summary = {"tesseract": tess_final, "gemini": gem_final}
    st.write(summary)

    if simulate_gate:
        allowed, matched = decide_gate(allowed_set, [summary["gemini"], summary["tesseract"]])
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
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Resetar estacionamento (esvaziar tudo)"):
                parking_spots = [None] * 30
                save_parking_spots(parking_spots)
                st.success("Estacionamento resetado.")
        with c2:
            if st.button("Liberar vaga aleatória (saída)"):
                parking_spots, freed = free_random_occupied(parking_spots)
                save_parking_spots(parking_spots)
                if freed is None:
                    st.warning("Não há vagas ocupadas para liberar.")
                else:
                    st.info(f"Vaga liberada: V{freed+1:02d}")
        with c3:
            st.caption("Luzes: 🟢 livre | 🔴 ocupada")

    return parking_spots

