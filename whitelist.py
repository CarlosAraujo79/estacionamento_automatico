from __future__ import annotations

import json
from typing import Optional

from utils import get_env_path, norm_plate


def db_path() -> str:
    return get_env_path("PLATES_DB_PATH", "plates_db.json")


def load_allowed_plates() -> list[str]:
    path = db_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("allowed_plates", [])
        if not isinstance(raw, list):
            return []
        return sorted({norm_plate(x) for x in raw if isinstance(x, str) and norm_plate(x)})
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_allowed_plates(plates: list[str]) -> None:
    path = db_path()
    plates = sorted({norm_plate(p) for p in plates if norm_plate(p)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"allowed_plates": plates}, f, ensure_ascii=False, indent=2)


def decide_gate(allowed: set[str], candidates: list[str]) -> tuple[bool, str]:
    for c in candidates:
        cc = norm_plate(c)
        if cc and cc in allowed:
            return True, cc
    return False, norm_plate(candidates[0]) if candidates else ""

