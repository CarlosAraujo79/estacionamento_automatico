from __future__ import annotations

import json
import random
from typing import Optional

import streamlit as st

from utils import get_env_path, norm_plate


def parking_path() -> str:
    return get_env_path("PARKING_STATE_PATH", "parking_state.json")


def load_parking_spots() -> list[Optional[str]]:
    path = parking_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        spots = data.get("spots", [])
        if not isinstance(spots, list) or len(spots) != 30:
            return [None] * 30
        out: list[Optional[str]] = []
        for s in spots:
            if s is None:
                out.append(None)
            elif isinstance(s, str):
                out.append(norm_plate(s) or None)
            else:
                out.append(None)
        return out
    except FileNotFoundError:
        return [None] * 30
    except Exception:
        return [None] * 30


def save_parking_spots(spots: list[Optional[str]]) -> None:
    path = parking_path()
    norm: list[Optional[str]] = []
    for s in spots[:30]:
        if s is None:
            norm.append(None)
        else:
            p = norm_plate(s)
            norm.append(p or None)
    while len(norm) < 30:
        norm.append(None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"spots": norm}, f, ensure_ascii=False, indent=2)


def occupy_random_spot(spots: list[Optional[str]], plate: str) -> tuple[list[Optional[str]], Optional[int]]:
    empty = [i for i, s in enumerate(spots) if not s]
    if not empty:
        return spots, None
    idx = random.choice(empty)
    spots2 = list(spots)
    spots2[idx] = norm_plate(plate) or plate
    return spots2, idx


def free_random_occupied(spots: list[Optional[str]]) -> tuple[list[Optional[str]], Optional[int]]:
    occ = [i for i, s in enumerate(spots) if s]
    if not occ:
        return spots, None
    idx = random.choice(occ)
    spots2 = list(spots)
    spots2[idx] = None
    return spots2, idx


def render_parking_grid(spots: list[Optional[str]]):
    st.subheader("Estacionamento (30 vagas: 3 fileiras × 10)")
    occupied = sum(1 for s in spots if s)
    st.caption(f"Ocupadas: {occupied}/30 | Livres: {30 - occupied}/30")

    for row in range(3):
        cols = st.columns(10)
        for col in range(10):
            i = row * 10 + col
            plate = spots[i]
            label = f"V{(i+1):02d}"
            if plate:
                cols[col].markdown(f"🔴 **{label}**  \n`{plate}`")
            else:
                cols[col].markdown(f"🟢 **{label}**  \nLivre")

