from __future__ import annotations

import os
from typing import Optional


def norm_plate(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.upper() if ch.isalnum())


def get_env_path(var: str, default: str) -> str:
    return os.getenv(var, default)


_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

# Possíveis conversões letra->dígito, com custo (menor = mais provável).
# Inclui alguns "atalhos" para erros comuns no OCR (ex.: 'G' às vezes vira '0').
_LETTER_TO_DIGIT_OPTIONS: dict[str, list[tuple[str, int]]] = {
    "O": [("0", 1)],
    "Q": [("0", 2)],
    "D": [("0", 2)],
    "I": [("1", 1)],
    "L": [("1", 2)],
    "Z": [("2", 1)],
    "S": [("5", 1)],
    "G": [("6", 1), ("0", 2)],  # preferir 6, mas permitir 0 (erro comum do OCR)
    "B": [("8", 1)],
    "A": [("4", 2)],
    "T": [("7", 2)],
}


def _letter_options(ch: str) -> list[tuple[str, int]]:
    ch = ch.upper()
    if ch.isalpha():
        return [(ch, 0)]
    if ch.isdigit():
        mapped = _DIGIT_TO_LETTER.get(ch)
        if mapped:
            return [(mapped, 1)]
        return [(ch, 6)]
    return [(ch, 10)]


def _digit_options(ch: str) -> list[tuple[str, int]]:
    ch = ch.upper()
    if ch.isdigit():
        return [(ch, 0)]
    if ch.isalpha():
        opts = _LETTER_TO_DIGIT_OPTIONS.get(ch)
        if opts:
            return opts
        return [(ch, 10)]
    return [(ch, 10)]


def _apply_pattern(candidate7: str, pattern: str) -> tuple[str, int]:
    """
    pattern: string de tamanho 7 com 'A' (letra) e '0' (dígito).
    Retorna (placa_corrigida, custo). Custo menor = melhor.
    """
    out = []
    cost = 0
    for ch, p in zip(candidate7.upper(), pattern):
        if p == "A":
            fixed, add = min(_letter_options(ch), key=lambda t: t[1])
            cost += add
            if not fixed.isalpha():
                cost += 5
            out.append(fixed)
        else:
            fixed, add = min(_digit_options(ch), key=lambda t: t[1])
            cost += add
            if not fixed.isdigit():
                cost += 5
            out.append(fixed)
    plate = "".join(out)
    return plate, cost


def interpret_plate(raw: Optional[str]) -> str:
    """
    Interpreta o OCR assumindo que a placa só pode ser:
    - AAA0000 (padrão antigo)
    - AAA0A00 (Mercosul)

    Retorna a melhor placa corrigida (7 chars) ou "".
    """
    s = norm_plate(raw)
    if not s:
        return ""

    # Considera janelas de tamanho 7 caso o OCR traga lixo extra.
    # Caso típico: o OCR duplica 1 char (ex.: 'I' -> 'I1'), gerando 8 chars.
    windows = [s[i : i + 7] for i in range(0, max(1, len(s) - 6))]
    if len(s) == 8:
        windows.extend([s[:i] + s[i + 1 :] for i in range(8)])
    best_plate = ""
    best_cost = 10**9

    patterns = ["AAA0000", "AAA0A00"]
    for w in windows:
        if len(w) != 7:
            continue
        for pat in patterns:
            plate, cost = _apply_pattern(w, pat.replace("0", "0"))
            # Validação final: 7 chars alfanuméricos e posições coerentes
            if len(plate) != 7 or not plate.isalnum():
                continue
            ok = True
            for ch, p in zip(plate, pat):
                if p == "A" and not ch.isalpha():
                    ok = False
                    break
                if p == "0" and not ch.isdigit():
                    ok = False
                    break
            if not ok:
                continue
            if cost < best_cost:
                best_cost = cost
                best_plate = plate

    return best_plate


def _raw_candidates_7(s: str) -> list[str]:
    """
    Gera candidatos de tamanho 7 a partir de uma string normalizada.
    - janelas de 7 em strings >= 7
    - caso típico de duplicação (len==8): remove 1 char em todas as posições
    """
    if not s:
        return []
    out = [s[i : i + 7] for i in range(0, max(1, len(s) - 6)) if len(s[i : i + 7]) == 7]
    if len(s) == 8:
        out.extend([s[:i] + s[i + 1 :] for i in range(8)])
    # dedup preservando ordem
    seen = set()
    uniq = []
    for x in out:
        if x not in seen and len(x) == 7:
            seen.add(x)
            uniq.append(x)
    return uniq


def _positional_cost(expected: str, observed: str) -> int:
    expected = expected.upper()
    observed = observed.upper()
    if expected.isalpha():
        for cand, c in _letter_options(observed):
            if cand == expected:
                return c
        return 20
    if expected.isdigit():
        for cand, c in _digit_options(observed):
            if cand == expected:
                return c
        return 20
    return 20


def best_whitelist_match(raw: Optional[str], allowed_set: set[str], max_cost: int = 2) -> str:
    """
    Tenta casar o OCR com uma placa da whitelist usando custos de confusão.
    Retorna a placa da whitelist (7 chars) se houver match confiável; caso contrário "".
    """
    s = norm_plate(raw)
    if not s or not allowed_set:
        return ""

    cands = _raw_candidates_7(s)
    if not cands:
        return ""

    best_plate = ""
    best_cost = 10**9

    for plate in allowed_set:
        if not isinstance(plate, str):
            continue
        p = norm_plate(plate)
        if len(p) != 7:
            continue
        for cand in cands:
            cost = 0
            for pe, ob in zip(p, cand):
                cost += _positional_cost(pe, ob)
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost = cost
                best_plate = p

    return best_plate if best_plate and best_cost <= max_cost else ""

