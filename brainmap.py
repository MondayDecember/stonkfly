#!/usr/bin/env python3
"""Помощник для 3D-страницы: живая карта мозга мухи.

Только читает. Один раз выгружает координаты тел нейронов (somaLocation из
MaleCNS), затем после каждого тика берёт из свежего чекпоинта мозга
(runs/paper/brain-*.npz, поле counts) число спайков каждого нейрона.
Код и состояние симуляции не меняются.

Файлы для страницы (runs/brain/):
  pos.bin   float32 [N,3]  координаты, нормированы в [-1,1], ось Y = дорсально
  cls.bin   uint8   [N]    код суперкласса (список в meta.json)
  flag.bin  uint8   [N]    1=KC 2=PAM11 4=PPL101 8=DNp20 16=DNpe017
  act.bin   uint8   [N]    активность за последний тик, log1p(спайки)*64
  meta.json, act.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/paper")
OUT = Path("runs/brain")
OUT.mkdir(parents=True, exist_ok=True)


def atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def xyz_of(v):
    try:
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        return a[:3] if a.size >= 3 else None
    except Exception:
        return None


def build():
    from stonkfly.neural.common import GRAPH, annotations

    with np.load(GRAPH, allow_pickle=False) as g:
        ids = g["ids"]
    a = annotations(ids)
    if "somaLocation" not in a.columns:
        sys.exit("В annotations.feather нет somaLocation. Колонки: " + ", ".join(a.columns))
    xyz = np.full((len(ids), 3), np.nan, np.float32)
    for i, v in enumerate(a["somaLocation"].to_numpy()):
        p = None if v is None else xyz_of(v)
        if p is not None:
            xyz[i] = p
    idx = np.flatnonzero(np.isfinite(xyz).all(1)).astype(np.int32)
    p = xyz[idx]
    center = (p.min(0) + p.max(0)) / 2
    scale = (p.max(0) - p.min(0)).max() / 2
    p = (p - center) / scale
    p = np.stack([p[:, 0], -p[:, 1], p[:, 2]], 1).astype(np.float32)

    types = a["type"].fillna("").astype(str).to_numpy()[idx]
    sup = a["superclass"].fillna("other").astype(str).to_numpy()[idx]
    classes = sorted(set(sup))
    code = np.array([classes.index(s) for s in sup], np.uint8)
    flag = np.zeros(len(idx), np.uint8)
    flag[np.char.startswith(types.astype(str), "KC")] |= 1
    flag[types == "PAM11"] |= 2
    flag[types == "PPL101"] |= 4
    flag[types == "DNp20"] |= 8
    flag[types == "DNpe017"] |= 16

    atomic(OUT / "pos.bin", p.tobytes())
    atomic(OUT / "cls.bin", code.tobytes())
    atomic(OUT / "flag.bin", flag.tobytes())
    np.save(OUT / "idx.npy", idx)
    meta = {
        "positioned": int(len(idx)),
        "total": int(len(ids)),
        "classes": classes,
        "groups": {k: int(((flag & b) > 0).sum()) for k, b in
                   [("KC", 1), ("PAM11", 2), ("PPL101", 4), ("DNp20", 8), ("DNpe017", 16)]},
    }
    atomic(OUT / "meta.json", json.dumps(meta).encode())
    print(f"карта мозга: {len(idx)} из {len(ids)} нейронов с координатами", flush=True)
    return idx


def main():
    idx = np.load(OUT / "idx.npy") if (OUT / "idx.npy").exists() and (OUT / "meta.json").exists() else build()
    last = None
    while True:
        cps = sorted(RUN.glob("brain-*.npz"), key=lambda q: q.stat().st_mtime)
        if cps:
            cp = cps[-1]
            m = cp.stat().st_mtime
            if m != last:
                try:
                    with np.load(cp, allow_pickle=False) as z:
                        counts = z["counts"]
                except Exception:
                    time.sleep(1)
                    continue
                c = counts[idx].astype(np.float32)
                atomic(OUT / "act.bin", np.minimum(255, np.round(np.log1p(c) * 64)).astype(np.uint8).tobytes())
                try:
                    tick = json.loads((RUN / "latest.json").read_text())["tick"]
                except Exception:
                    tick = None
                atomic(OUT / "act.json", json.dumps({
                    "mtime": m, "tick": tick,
                    "active": int((c > 0).sum()), "spikes": int(c.sum()),
                }).encode())
                last = m
        time.sleep(2)


if __name__ == "__main__":
    main()
