#!/usr/bin/env python3
"""Flappy Fly: настоящий мозг мухи (MaleCNS, тот же, что у торговой мухи) играет во Flappy Bird.

Сезон 2 (mb-v1): взмах решает сама память мухи.
Каждый шаг игры муха видит кадр 320x180, мозг живёт STEP_MS миллисекунд своего времени (~1 с счёта).
Взмах читается с выходных нейронов грибовидного тела, тех самых, чьи входные синапсы учатся:
MBON11 (отсек γ1) «за взмах», MBON07 (отсек α1) «против». Оценка = (за − против) по частоте спайков
и мембранному потенциалу, взмах — когда оценка в верхних 20% за последние 200 шагов
(столько взмахов нужно, чтобы вообще держаться в воздухе: G·dt/|FLAP_V| ≈ 0.22).
Сигнал ошибки после удара, по направлению:
  удар снизу (пол или нижняя труба)  → дофамин PAM11 в отсек α1: ослабляются синапсы KC→MBON07
                                        для картинок последней секунды → в таких местах махать чаще;
  удар сверху (верхняя труба)        → PPL101 в отсек γ1: ослабляются KC→MBON11 → махать реже.
Правило обучения то же, что у торговой мухи (Huang, Luo et al. 2024). После удара — 0,8 с тёмного
экрана проигрыша, чтобы след дофамина не ложился на начало следующей попытки.
В сезоне 1 (easy-v3) взмах решали DNp20, до которых выученное не доходило (stonkfly check).

Пишет для страницы flappy.html в runs/flappy/: state.json, attempts.jsonl, eye.png, brain.npz.
Пауза: http://<IP>:8082/pause?s=15 (не дольше 30 с) и /resume. 3D-страница ставит игру на паузу,
когда муха отходит от компьютера закрыть дверь. На паузе игра и мозг стоят, счёт шагов не идёт.
Запуск: .venv/bin/python flappyfly.py   (stonkfly start запускает сам)
"""
import json
import os
import signal
import sys
import threading
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
PORT = int(os.environ.get("FLAPPY_PORT", 8082))
PAUSE = {"until": 0.0}
LEVEL = "mb-v1"   # сезон 2: взмах решает выход памяти (MBON), ошибка по направлению удара
GAMEOVER_MS = 800   # тёмный экран после удара, мс времени мозга


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
        """Возвращает (прошёл_трубу, разбился). Куда ударился — в self.crash: "low" (пол, нижняя труба) или "high"."""
        self.steps += 1
        self.crash = None
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
        if crashed:
            self.crash = "low"
        for x, gy in self.pipes:
            if x - BIRD_R < BIRD_X < x + PIPE_W + BIRD_R and not (gy - GAP / 2 + BIRD_R < self.y < gy + GAP / 2 - BIRD_R):
                crashed = True
                self.crash = self.crash or ("low" if self.y > gy else "high")
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


def control_server():
    """Маленький сервер паузы: /pause?s=15, /resume, /state. Отвечает всем (CORS), пауза максимум 30 с."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/pause":
                try:
                    sec = float(parse_qs(u.query).get("s", ["15"])[0])
                except ValueError:
                    sec = 15.0
                PAUSE["until"] = time.time() + min(30.0, max(0.0, sec))
            elif u.path == "/resume":
                PAUSE["until"] = 0.0
            body = json.dumps({"paused": PAUSE["until"] > time.time()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    try:
        ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
    except OSError as e:
        print(f"пауза недоступна (порт {PORT}): {e}", flush=True)


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


class Readout:
    """Взмах читается с выхода памяти: MBON11 (γ1) за, MBON07 (α1) против.
    Частоты спайков за 80 мс идут ступеньками, поэтому к ним добавлен мембранный потенциал тех же клеток.
    Каждый признак нормируется по своей скользящей средней и разбросу (за ~500 шагов)."""

    KEYS = ("pro_hz", "pro_mv", "anti_hz", "anti_mv")

    def __init__(self, brain, seconds):
        mb = np.asarray(brain.circuit["mb"])
        if len(mb) != 6:
            raise ValueError("ожидались 4 MBON07 и 2 MBON11")
        self.b, self.sec = brain, seconds
        self.anti, self.pro = mb[:4], mb[4:]    # identify(): сначала MBON07 (награда), потом MBON11 (наказание)
        self.stats = {}

    def raw(self):
        c, v = self.b.counts, self.b.v
        return {"pro_hz": float(c[self.pro].mean() / self.sec), "pro_mv": float(v[self.pro].mean()),
                "anti_hz": float(c[self.anti].mean() / self.sec), "anti_mv": float(v[self.anti].mean())}

    def z(self, r, learn=True):
        out = {}
        for k in self.KEYS:
            x = r[k]
            m, var, n = self.stats.get(k, (x, 1.0, 0))
            if learn:
                a = max(1 / (n + 1), 1 / 500)
                m += a * (x - m)
                var += a * ((x - m) ** 2 - var)
                self.stats[k] = (m, var, n + 1)
            out[k] = float(np.clip((x - m) / np.sqrt(var + 1e-6), -4, 4))
        return out

    def score(self, r, n, learn=True):
        z = self.z(r, learn)
        pro, anti = z["pro_hz"] + z["pro_mv"], z["anti_hz"] + z["anti_mv"]
        return pro - anti + 1e-4 * decision_score(n), pro, anti


def decision_score(n):
    """Сезон 1 и разбор равенств: чем сильнее правый DNp20 относительно левого, тем выше. Частоты за 80 мс идут ступеньками,
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
    params = {"level": LEVEL, "G": G, "FLAP_V": FLAP_V, "SPEED": SPEED, "SPACING": SPACING, "GAP": GAP, "STEP_MS": STEP_MS, "FLAP_RATE": FLAP_RATE,
              "readout": "MBON11-MBON07", "teach": "crash-direction", "gameover_ms": GAMEOVER_MS}
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
    game = Game(np.random.default_rng())
    readout = Readout(ctl.brain, STEP_MS / 1000)
    dark = np.zeros((H, W, 3), np.uint8)
    from collections import deque
    recent = deque(maxlen=200)

    def save(*_):
        ctl.save(ckpt)
        if _:
            print("Flappy Fly: память сохранена, выхожу", flush=True)
            sys.exit(0)

    signal.signal(signal.SIGINT, save)
    signal.signal(signal.SIGTERM, save)
    threading.Thread(target=control_server, daemon=True).start()
    print(f"Flappy Fly: играю (пауза: порт {PORT})", flush=True)
    last, was_paused = None, False
    while True:
        if PAUSE["until"] > time.time():   # муха отошла от компьютера: игра и мозг стоят
            if not was_paused:
                print("пауза", flush=True)
                was_paused = True
            if last:
                atomic(OUT / "state.json", json.dumps({**last, "wall": time.time(), "paused": True, "flap": False, "passed": False, "crashed": False}).encode())
            time.sleep(0.25)
            continue
        if was_paused:
            print("продолжаю", flush=True)
            was_paused = False
        frame = game.frame()
        n = ctl.observe(frame, "none")
        d = n["difference_hz"]
        raw = readout.raw()
        sc, pro, anti = readout.score(raw, n)
        if len(recent) >= 20:      # средний ранг: равные значения не «съедают» взмахи
            arr = np.asarray(recent)
            frac = ((arr < sc).sum() + 0.5 * (arr == sc).sum()) / len(arr)
            flap = frac >= 1 - FLAP_RATE
        else:
            flap = sc > float(np.median(recent or [sc]))
        recent.append(sc)
        thr = float(np.percentile(recent, 100 * (1 - FLAP_RATE)))
        passed, crashed = game.step(flap, STEP_MS / 1000)
        best = max(best, game.score)
        tail = attempts[-20:]
        st_ = sum(a["steps"] for a in tail) + game.steps
        rate_now = (sum(a["flaps"] for a in tail) + game.flaps) / st_ if st_ else FLAP_RATE
        state = {
            "wall": time.time(), "attempt": len(attempts) + 1, "score": game.score, "best": best, "steps": game.steps,
            "y": game.y, "vy": game.vy, "pipes": game.pipes, "flap": bool(flap), "passed": passed, "crashed": crashed,
            "left_hz": n["left_hz"], "right_hz": n["right_hz"], "diff_hz": d, "gate": n["gate_spikes"],
            "decision": sc, "threshold": thr, "pro": pro, "anti": anti, **raw, "teach": game.crash,
            "spikes": n["total_spikes"], "compute": n["compute_seconds"], "changed_edges": n["memory"]["changed_edges"],
            "mean_efficacy": n["memory"]["mean_efficacy"], "baseline": base_at(rate_now), "flap_rate": rate_now, "level": LEVEL,
            "paused": False, "world": {"W": W, "H": H, "GROUND": GROUND, "GAP": GAP, "PIPE_W": PIPE_W, "BIRD_X": BIRD_X, "BIRD_R": BIRD_R, "STEP_MS": STEP_MS},
        }
        Image.fromarray(frame).save(OUT / "eye.tmp.png")
        (OUT / "eye.tmp.png").replace(OUT / "eye.png")
        if crashed:
            atomic(OUT / "state.json", json.dumps(state).encode())
            # сигнал ошибки по направлению удара, на кадре самого удара
            ctl.observe(game.frame(), "reward" if game.crash == "low" else "aversive")
            ctl.brain.rgb_step(dark, GAMEOVER_MS, learning=True)   # экран проигрыша
            rec = {"attempt": len(attempts) + 1, "score": game.score, "steps": game.steps, "flaps": game.flaps, "crash": game.crash, "wall": time.time()}
            attempts.append(rec)
            with hist_path.open("a") as h:
                h.write(json.dumps(rec) + "\n")
            print(f"попытка {rec['attempt']}: {rec['score']} труб, {rec['steps']} шагов, удар {'снизу' if game.crash == 'low' else 'сверху'}, рекорд {best}", flush=True)
            if len(attempts) % 10 == 0:
                save()
            game.reset()
        last = state
        atomic(OUT / "state.json", json.dumps(state).encode())


if __name__ == "__main__":
    main()
