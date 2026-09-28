#!/usr/bin/env python3
"""Муха выбирает цвет: обучение, для которого грибовидное тело мухи и существует.

Как опыты с живыми дрозофилами (Т-лабиринт, Tully & Quinn 1985; цвет — Schnaitmann et al. 2010):
  УРОК     муха смотрит на цвет, и в этот момент ей дают сахар (дофамин PAM11 → отсек α1)
           или удар (PPL101 → отсек γ1). Уроки чередуются: сладкий цвет, горький цвет.
  ЭКЗАМЕН  муха по очереди смотрит на оба цвета (порядок случайный), без сахара и удара.
           Выбор читается с выхода памяти: «тянет» = MBON11 (γ1) − MBON07 (α1), по спайкам и потенциалу.
           Какой цвет тянет сильнее, тот и выбран.
Сахар ослабляет входы MBON07 для сладкого цвета, удар ослабляет входы MBON11 для горького, и оба урока
толкают выбор к сладкому. Никакой подсказки на экзамене нет: выбор делает только память.

Честные проверки:
  * сначала PRETEST экзаменов без уроков: врождённое предпочтение мухи;
  * каждые REVERSAL экзаменов сладкий и горький цвета меняются местами. Врождённая любовь к синему
    не может следовать за переменой, а обучение может. Главная цифра — средняя точность по обоим
    вариантам (сладкий синий и сладкий зелёный), случайность = 50%.

Пишет в runs/colors/: state.json, exams.jsonl, eye.png, brain.npz. Запуск: .venv/bin/python colors.py
"""
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from flappyfly import Readout, atomic   # тот же выход памяти, что во Flappy сезона 2

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = Path("runs/colors")
OUT.mkdir(parents=True, exist_ok=True)

STEP_MS = 80.0
ITI_MS = float(os.environ.get("COLORS_ITI_MS", 1000))      # серый экран между пробами: след дофамина успевает угаснуть
PRETEST = int(os.environ.get("COLORS_PRETEST", 20))
REVERSAL = int(os.environ.get("COLORS_REVERSAL", 150))
LEVEL = "colors-v1"
W, H = 320, 180
BG = (70, 70, 70)
COLORS = {"blue": (40, 90, 255), "green": (40, 210, 60)}
RU = {"blue": "синий", "green": "зелёный"}


def frame(color, rng):
    """Серый фон и большой цветной круг примерно посередине (чуть сдвигается, как при живом взгляде)."""
    f = np.empty((H, W, 3), np.uint8)
    f[:] = BG
    if color:
        cx, cy = W / 2 + rng.uniform(-12, 12), H / 2 + rng.uniform(-8, 8)
        yy, xx = np.ogrid[:H, :W]
        f[(xx - cx) ** 2 + (yy - cy) ** 2 <= 62 ** 2] = COLORS[color]
    return f


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summary(exams):
    pre = [e for e in exams if e["phase"] == "pre"]
    tr = [e for e in exams if e["phase"] == "train"]
    s = {"exams": len(exams), "pretest": len(pre)}
    if pre:
        s["pre_blue"] = sum(e["choice"] == "blue" for e in pre) / len(pre)   # врождённая любовь к синему
    last = tr[-100:]
    if last:
        k = sum(e["correct"] for e in last)
        s["last_n"], s["last_acc"] = len(last), k / len(last)
        s["last_ci"] = wilson(k, len(last))
        by = {}
        for c in COLORS:
            part = [e for e in last if e["sweet"] == c]
            if part:
                by[c] = sum(e["correct"] for e in part) / len(part)
        s["by_sweet"] = by
        if len(by) == 2:
            s["balanced"] = sum(by.values()) / 2
    return s


def main():
    from PIL import Image

    from stonkfly.config import Settings
    from stonkfly.neural.controller import FlyController

    print("Муха выбирает цвет: загружаю мозг мухи…", flush=True)
    ctl = FlyController(Settings(learning=True, neural_ms=STEP_MS, pulse_ms=STEP_MS))
    b = ctl.brain
    params = {"level": LEVEL, "colors": COLORS, "bg": BG, "iti_ms": ITI_MS, "pretest": PRETEST, "reversal": REVERSAL,
              "readout": "MBON11-MBON07"}
    old = json.loads((OUT / "params.json").read_text()) if (OUT / "params.json").exists() else None
    if old != json.loads(json.dumps(params)):   # правила поменялись: старое в архив, честный старт с нуля
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("exams.jsonl", "brain.npz"):
            if (OUT / name).exists():
                (OUT / name).rename(OUT / f"{name.split('.')[0]}-{stamp}.{name.split('.')[1]}")
        (OUT / "params.json").write_text(json.dumps(params))
        print("новые правила: старая история и память отложены в архив", flush=True)
    ckpt = OUT / "brain.npz"
    if ckpt.exists():
        try:
            ctl.restore(ckpt)
            print("память восстановлена", flush=True)
        except Exception as e:
            print("не удалось восстановить память, начинаю заново:", e, flush=True)
    path = OUT / "exams.jsonl"
    exams = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    readout = Readout(b, STEP_MS / 1000)
    rng = np.random.default_rng()
    gray = frame(None, rng)
    lesson_i = len(exams)

    def save(*_):
        ctl.save(ckpt)
        if _:
            print("память сохранена, выхожу", flush=True)
            sys.exit(0)

    signal.signal(signal.SIGINT, save)
    signal.signal(signal.SIGTERM, save)

    def sweet_now():
        trained = max(0, len(exams) - PRETEST)
        return "blue" if (trained // REVERSAL) % 2 == 0 else "green"

    def publish(**kw):
        st = {"wall": time.time(), "level": LEVEL, "sweet": sweet_now(), "phase_exam": "pre" if len(exams) < PRETEST else "train",
              "reversal_every": REVERSAL, "pretest": PRETEST, "summary": summary(exams), "last": exams[-1] if exams else None,
              "changed_edges": b.memory()["changed_edges"], **kw}
        atomic(OUT / "state.json", json.dumps(st, ensure_ascii=False).encode())

    def look(color, stim="none"):
        f = frame(color, rng)
        n = ctl.observe(f, stim)
        Image.fromarray(f).save(OUT / "eye.tmp.png")
        (OUT / "eye.tmp.png").replace(OUT / "eye.png")
        return n

    def pause():
        publish(step="pause")
        look(None)
        b.rgb_step(gray, ITI_MS - STEP_MS, learning=True)

    print(f"играю: {PRETEST} экзаменов без уроков, потом урок-экзамен, смена сладкого цвета каждые {REVERSAL}", flush=True)
    while True:
        # ---- экзамен ----
        sweet = sweet_now()
        order = list(rng.permutation(list(COLORS)))
        pull = {}
        for i, c in enumerate(order):
            publish(step="exam", looking=c, order=order, pull=pull)
            look(c)
            n = look(c)            # второй кадр: реакция устоялась
            s, pro, anti = readout.score(readout.raw(), n)
            pull[c] = {"pull": s, "pro": pro, "anti": anti}
            if i == 0:
                look(None)
        choice = max(order, key=lambda c: pull[c]["pull"])
        phase = "pre" if len(exams) < PRETEST else "train"
        rec = {"n": len(exams) + 1, "phase": phase, "sweet": sweet, "order": order, "choice": choice,
               "correct": choice == sweet, "pull": {c: round(pull[c]["pull"], 3) for c in order}, "wall": time.time()}
        exams.append(rec)
        with path.open("a") as h:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sm = summary(exams)
        print(f"экзамен {rec['n']} ({'до уроков' if phase == 'pre' else 'сладкий ' + RU[sweet]}): выбрала {RU[choice]}"
              + ("" if phase == "pre" else " ✓" if rec["correct"] else " ✗")
              + (f"  | последние {sm['last_n']}: {sm['last_acc'] * 100:.0f}%" if "last_acc" in sm else ""), flush=True)
        publish(step="choice", order=order, pull=pull, choice=choice)
        if len(exams) % 20 == 0:
            save()
        pause()
        if len(exams) < PRETEST:
            continue
        # ---- урок: по очереди сладкий цвет с сахаром и горький с ударом ----
        sweet = sweet_now()
        bitter = "green" if sweet == "blue" else "blue"
        kind = "sugar" if lesson_i % 2 == 0 else "shock"
        lesson_i += 1
        c = sweet if kind == "sugar" else bitter
        publish(step="lesson", color=c, kind=kind)
        look(c)
        look(c, "reward" if kind == "sugar" else "aversive")
        pause()


if __name__ == "__main__":
    main()
