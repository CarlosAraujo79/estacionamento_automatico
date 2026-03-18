from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple


# Evita ruído no console em ambientes com múltiplas instalações do Matplotlib.
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)

def _need(pkg: str, hint: str):
    raise RuntimeError(f"Dependência ausente: {pkg}. {hint}")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        _need("opencv-python", "Instale com: pip install -r requirements.txt")
    return cv2


def _import_np():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        _need("numpy", "Instale com: pip install -r requirements.txt")
    return np


def _import_pil_image():
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        _need("Pillow", "Instale com: pip install -r requirements.txt")
    return Image


@dataclass(frozen=True)
class Detection:
    xyxy: Tuple[int, int, int, int]
    conf: float
    cls: int


def _load_image_bgr(path: str) -> np.ndarray:
    cv2 = _import_cv2()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não consegui ler a imagem: {path}")
    return img


def _clip_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _pad_xyxy(x1: int, y1: int, x2: int, y2: int, pad_ratio: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    return _clip_xyxy(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)


def detect_best_plate(model_path: str, image_bgr: np.ndarray, conf: float, device: str) -> Detection:
    np = _import_np()
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Falha ao importar ultralytics. Rode: pip install -r requirements.txt"
        ) from e

    model = YOLO(model_path)
    results = model.predict(source=image_bgr, conf=conf, device=device, verbose=False)
    if not results:
        raise RuntimeError("Sem resultados do modelo.")

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("Nenhuma placa detectada (nenhuma bbox acima do conf).")

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)

    best_i = int(np.argmax(confs))
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy[best_i].tolist()]
    return Detection(xyxy=(x1, y1, x2, y2), conf=float(confs[best_i]), cls=int(clss[best_i]))


def preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    np = _import_np()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return thr


def preprocess_variants_for_ocr(crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """
    Retorna várias versões pré-processadas para aumentar a chance de OCR,
    especialmente em placas antigas / com baixo contraste.
    """
    cv2 = _import_cv2()
    np = _import_np()

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    variants: dict[str, np.ndarray] = {}

    # 1) Base atual (adaptive + close)
    variants["adaptive_gauss_close"] = preprocess_for_ocr(crop_bgr)

    # 2) Otsu (bom quando há contraste razoável)
    g2 = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["otsu"] = otsu

    # 3) Otsu invertido
    variants["otsu_inv"] = cv2.bitwise_not(otsu)

    # 4) Adaptive mean (às vezes melhor que gauss)
    am = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)
    variants["adaptive_mean"] = am

    # 5) Sharpen + Otsu (ajuda em fontes "apagadas")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    sharp = cv2.GaussianBlur(sharp, (3, 3), 0)
    _, sharp_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["sharpen_otsu"] = sharp_otsu

    # 6) Denoise + adaptive
    den = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    den_ad = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants["denoise_adaptive"] = den_ad

    # 7) Morfologia para "engrossar" caracteres (placas gastas)
    dil = cv2.dilate(variants["adaptive_gauss_close"], np.ones((2, 2), np.uint8), iterations=1)
    variants["adaptive_dilate"] = dil

    # 8) Morfologia para "afinar" caracteres (placas com borrão)
    er = cv2.erode(variants["adaptive_gauss_close"], np.ones((2, 2), np.uint8), iterations=1)
    variants["adaptive_erode"] = er

    return variants


def ocr_tesseract(img: np.ndarray, psm: int = 7, lang: str = "eng") -> str:
    try:
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "Falha ao importar pytesseract. Rode: pip install -r requirements.txt"
        ) from e

    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return "".join(ch for ch in text.upper() if ch.isalnum())


def ocr_gemini(crop_bgr: np.ndarray, api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    cv2 = _import_cv2()
    Image = _import_pil_image()
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(
            "Falha ao importar google-generativeai. Rode: pip install -r requirements.txt"
        ) from e

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)

    prompt = (
        "Extraia APENAS o texto da placa do veículo da imagem. "
        "Responda somente com o texto (sem espaços, sem hífen, sem explicações). "
        "Use letras maiúsculas."
    )
    resp = model.generate_content([prompt, pil])
    out = (getattr(resp, "text", "") or "").strip().upper()
    return "".join(ch for ch in out if ch.isalnum())


def draw_detection(image_bgr: np.ndarray, det: Detection) -> np.ndarray:
    cv2 = _import_cv2()
    x1, y1, x2, y2 = det.xyxy
    out = image_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        out,
        f"plate conf={det.conf:.2f}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Caminho da imagem de entrada")
    ap.add_argument("--model", default="plaquinhas.pt", help="Caminho do modelo YOLO (.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence mínimo para bbox")
    ap.add_argument("--device", default="cpu", help="cpu, 0, 0,1 ... (GPU se disponível)")
    ap.add_argument("--pad", type=float, default=0.08, help="Padding no crop (fração da bbox)")
    ap.add_argument("--ocr", choices=["tesseract", "gemini"], default="tesseract")
    ap.add_argument("--tess-lang", default="eng", help="Idioma do tesseract (ex: eng, por)")
    ap.add_argument("--tess-psm", type=int, default=7, help="PSM do tesseract (7 costuma funcionar bem)")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash", help="Nome do modelo Gemini")
    ap.add_argument("--gemini-api-key", default=None, help="Chave da API Gemini (ou use env GEMINI_API_KEY)")
    ap.add_argument("--outdir", default="outputs", help="Pasta para salvar resultados")
    ap.add_argument("--save-debug", action="store_true", help="Salva imagem com bbox + crops")
    ap.add_argument("--show", action="store_true", help="Mostra janelas com bbox/crop (requer ambiente com display)")
    args = ap.parse_args()

    img = _load_image_bgr(args.image)
    h, w = img.shape[:2]

    det = detect_best_plate(args.model, img, conf=args.conf, device=args.device)
    x1, y1, x2, y2 = _pad_xyxy(*det.xyxy, pad_ratio=args.pad, w=w, h=h)
    crop = img[y1:y2, x1:x2]

    ocr_text: Optional[str] = None
    pre: Optional["np.ndarray"] = None
    if args.ocr == "tesseract":
        pre = preprocess_for_ocr(crop)
        ocr_text = ocr_tesseract(pre, psm=args.tess_psm, lang=args.tess_lang)
    else:
        key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Defina --gemini-api-key ou a variável de ambiente GEMINI_API_KEY.")
        ocr_text = ocr_gemini(crop, api_key=key, model_name=args.gemini_model)

    should_save = args.save_debug

    if args.show:
        # Em ambientes headless (sem DISPLAY) o imshow tende a falhar.
        if not os.getenv("DISPLAY") and os.name != "nt":
            should_save = True
        else:
            cv2 = _import_cv2()
            boxed = draw_detection(img, det)
            try:
                cv2.imshow("plate detection (bbox)", boxed)
                cv2.imshow("plate crop", crop)
                if pre is not None:
                    cv2.imshow("plate crop (tesseract preproc)", pre)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                should_save = True

    os.makedirs(args.outdir, exist_ok=True)
    if should_save:
        cv2 = _import_cv2()
        boxed = draw_detection(img, det)
        cv2.imwrite(os.path.join(args.outdir, "boxed.jpg"), boxed)
        cv2.imwrite(os.path.join(args.outdir, "crop.jpg"), crop)
        if pre is not None:
            cv2.imwrite(os.path.join(args.outdir, "crop_preproc.png"), pre)

    print(ocr_text or "")


if __name__ == "__main__":
    main()
