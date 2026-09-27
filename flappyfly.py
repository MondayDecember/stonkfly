#!/usr/bin/env python3
"""Flappy Fly: настоящий мозг мухи (MaleCNS, тот же, что у торговой мухи) играет во Flappy Bird.

Каждый шаг игры муха видит кадр 320x180, мозг живёт FLAP_MS миллисекунд своего времени
(~1 с счёта на процессоре). Взмах = правый DNp20 стреляет сильнее левого заметнее, чем обычно
для этой мухи: разница R-L выше своего 80-го перцентиля за последние 200 шагов (FLAP_RATE=0.2).
Относительный порог нужен, потому что у мухи есть постоянный перекос вправо; когда махать,
решает мозг в ответ на то, что видит. Пройденная труба = дофамин (PAM11), столкновение = наказание (PPL101).
Синапсы памяти учатся так же, как у торговой мухи. Игра идёт во «времени мозга»:
для мухи это реальное время, для нас замедление примерно в 12 раз.

Пишет для страницы flappy.html в runs/flappy/: state.json, attempts.jsonl, eye.png, brain.npz.
Запуск: .venv/bin/python flappyfly.py   (stonkfly start запускает сам)
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = Path("runs/flappy")
OUT.mkdir(parents=True, exist_ok=True)

STEP_MS = float(os.environ.get("FLAP_MS", 80))        # сколько живёт мозг на один кадр
FLAP_RATE = float(os.environ.get("FLAP_RATE", 0.2))   # доля шагов со взмахом (подобрано на случайной игре)
W, H, GROUND = 320, 180, 16
# лёгкий режим: шире проход, медленнее трубы, мягче гравитация — больше шансов долететь и получить дофамин
G, FLAP_V, SPEED, SPACING, GAP, PIPE_W, BIRD_X, BIRD_R = 240.0, -88.0, 45.0, 210.0, 125.0, 34, 70, 7
LEVEL = "easy-v3"   # v3: равенства П−Л решает DNpe017, планка по реальной доле взмахов


def atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class Game:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.y, self.vy, self.score, self.steps, self.flaps = H / 2, 0.0, 0, 0, 0
        self.pipes = [[W - 60.0, self.gap()]]

    def gap(self):
        return float(self.rng.uniform(14 + GAP / 2, H - GROUND - 10 - GAP / 2))

    def step(self, flap, dt):
        """Возвращает (прошёл_трубу, разбился)."""
        self.steps += 1
        if flap:
            self.vy, self.flaps = FLAP_V, self.flaps + 1
        self.vy += G * dt
        self.y += self.vy * dt
        if self.y < BIRD_R:                       # потолок не убивает, просто упирается
            self.y, self.vy = BIRD_R, 0.0
        passed = False
        for p in self.pipes:
            old = p[0]
            p[0] -= SPEED * dt
            if old + PIPE_W >= BIRD_X > p[0] + PIPE_W:
                self.score += 1
                passed = True
        if self.pipes[-1][0] < W - SPACING:
            self.pipes.append([float(W), self.gap()])
        self.pipes = [p for p in self.pipes if p[0] > -PIPE_W]
        crashed = self.y + BIRD_R > H - GROUND
        for x, gy in self.pipes:
            if x - BIRD_R < BIRD_X < x + PIPE_W + BIRD_R and not (gy - GAP / 2 + BIRD_R < self.y < gy + GAP / 2 - BIRD_R):
                crashed = True
        return passed, crashed

    def frame(self):
        """То, что видит муха: контрастная картинка 320x180."""
        f = np.empty((H, W, 3), np.uint8)
        f[:] = (150, 205, 255)                                   # небо
        f[H - GROUND:] = (205, 180, 110)                         # земля
        for x, gy in self.pipes:
            a, b = int(max(0, x)), int(min(W, x + PIPE_W))
            if a < b:
                top, bot = int(gy - GAP / 2), int(gy + GAP / 2)
                f[:max(0, top), a:b] = (35, 135, 55)
                f[max(0, bot):H - GROUND, a:b] = (35, 135, 55)
                f[max(0, top - 6):max(0, top), a:b] = (20, 90, 35)
                f[max(0, bot):max(0, bot + 6), a:b] = (20, 90, 35)
        yy, xx = np.ogrid[:H, :W]
        f[(xx - BIRD_X) ** 2 + (yy - self.y) ** 2 <= BIRD_R ** 2] = (250, 215, 40)
        f[(xx - BIRD_X - 3) ** 2 + (yy - self.y + 2) ** 2 <= 2] = (0, 0, 0)
        return f


def random_baseline(rate, n=400, seed=1):
    """Сколько труб в среднем проходит муха, которая машет наугад с частотой rate. Планка для честного сравнения."""
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(n):
        g = Game(rng)
        while True:
            _, c = g.step(rng.random() < rate, STEP_MS / 1000)
            if c or g.steps > 5000:
                break
        total += g.score
    return total / n


RATES = np.round(np.arange(0.02, 0.52, 0.04), 2)


def decision_score(n):
    """Чем сильнее правый DNp20 относительно левого, тем выше. Частоты за 80 мс идут ступеньками,
    поэтому равенства разбирает нейрон DNpe017 (тот же «разрешающий», что у торговой мухи),
    а оставшиеся — общая активность DNp20. Всё это — выход мозга, без случайности."""
    return n["difference_hz"] + 0.01 * n["gate_spikes"] + 1e-4 * (n["left_hz"] + n["right_hz"])


def main():
    from PIL import Image

    from stonkfly.config import Settings
    from stonkfly.neural.controller import FlyController

    print("Flappy Fly: загружаю мозг мухи…", flush=True)
    settings = Settings(learning=True, neural_ms=STEP_MS, pulse_ms=min(200, STEP_MS))
    ctl = FlyController(settings)
    params = {"level": LEVEL, "G": G, "FLAP_V": FLAP_V, "SPEED": SPEED, "SPACING": SPACING, "GAP": GAP, "STEP_MS": STEP_MS, "FLAP_RATE": FLAP_RATE}
    old = json.loads((OUT / "params.json").read_text()) if (OUT / "params.json").exists() else None
    if old != params:   # правила игры поменялись: старую историю и память — в архив, честный старт с нуля
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("attempts.jsonl", "brain.npz"):
            if (OUT / name).exists():
                (OUT / name).rename(OUT / f"{name.split('.')[0]}-{stamp}.{name.split('.')[1]}")
        print("новые правила игры: старая история и память мухи отложены в архив", flush=True)
        (OUT / "params.json").write_text(json.dumps(params))
    table = [random_baseline(r) for r in RATES]
    base_at = lambda r: float(np.interp(r, RATES, table))
    print("планка случайного махания: " + ", ".join(f"{r:.2f}→{b:.2f}" for r, b in zip(RATES, table)), flush=True)
    ckpt = OUT / "brain.npz"
    if ckpt.exists():
        try:
            ctl.restore(ckpt)
            print("память восстановлена из прошлых попыток", flush=True)
        except Exception as e:
            print("не удалось восстановить память, начинаю заново:", e, flush=True)
    hist_path = OUT / "attempts.jsonl"
    attempts = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()] if hist_path.exists() else []
    best = max([a["score"] for a in attempts], default=0)
    game, stim = Game(np.random.default_rng()), "none"
    from collections import deque
    recent = deque(maxlen=200)

    def save(*_):
        ctl.save(ckpt)
        if _:
            print("Flappy Fly: память сохранена, выхожу", flush=True)
            sys.exit(0)

    signal.signal(signal.SIGINT, save)
    signal.signal(signal.SIGTERM, save)
    print("Flappy Fly: играю", flush=True)
    while True:
        frame = game.frame()
        n = ctl.observe(frame, stim)
        d = n["difference_hz"]
        sc = decision_score(n)
        if len(recent) >= 20:      # средний ранг: равные значения не «съедают» взмахи
            arr = np.asarray(recent)
            frac = ((arr < sc).sum() + 0.5 * (arr == sc).sum()) / len(arr)
            flap = frac >= 1 - FLAP_RATE
        else:
            flap = sc > float(np.median(recent or [sc]))
        recent.append(sc)
        thr = float(np.percentile(recent, 100 * (1 - FLAP_RATE)))
        passed, crashed = game.step(flap, STEP_MS / 1000)
        stim = "aversive" if crashed else "reward" if passed else "none"
        best = max(best, game.score)
        tail = attempts[-20:]
        st_ = sum(a["steps"] for a in tail) + game.steps
        rate_now = (sum(a["flaps"] for a in tail) + game.flaps) / st_ if st_ else FLAP_RATE
        state = {
            "wall": time.time(), "attempt": len(attempts) + 1, "score": game.score, "best": best, "steps": game.steps,
            "y": game.y, "vy": game.vy, "pipes": game.pipes, "flap": bool(flap), "passed": passed, "crashed": crashed,
            "left_hz": n["left_hz"], "right_hz": n["right_hz"], "diff_hz": d, "threshold_hz": thr, "gate": n["gate_spikes"],
            "spikes": n["total_spikes"], "compute": n["compute_seconds"], "changed_edges": n["memory"]["changed_edges"],
            "mean_efficacy": n["memory"]["mean_efficacy"], "baseline": base_at(rate_now), "flap_rate": rate_now, "level": LEVEL,
            "world": {"W": W, "H": H, "GROUND": GROUND, "GAP": GAP, "PIPE_W": PIPE_W, "BIRD_X": BIRD_X, "BIRD_R": BIRD_R, "STEP_MS": STEP_MS},
        }
        Image.fromarray(frame).save(OUT / "eye.tmp.png")
        (OUT / "eye.tmp.png").replace(OUT / "eye.png")
        if crashed:
            rec = {"attempt": len(attempts) + 1, "score": game.score, "steps": game.steps, "flaps": game.flaps, "wall": time.time()}
            attempts.append(rec)
            with hist_path.open("a") as h:
                h.write(json.dumps(rec) + "\n")
            print(f"попытка {rec['attempt']}: {rec['score']} труб, {rec['steps']} шагов, рекорд {best}", flush=True)
            if len(attempts) % 10 == 0:
                save()
            game.reset()
        atomic(OUT / "state.json", json.dumps(state).encode())


if __name__ == "__main__":
    main()
