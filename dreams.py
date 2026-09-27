#!/usr/bin/env python3
"""Сны мухи: раз в час состояние мозга и торговли превращается в описание сна, ComfyUI рисует картинку.

Из чего складывается сон (всё настоящее):
  * какие отделы мозга активнее всего в последнем тике (runs/brain: act.bin + классы нейронов);
  * что было за последний час торговли (runs/paper/history.jsonl): награды, наказания, баланс, цена BTC;
  * время и сезон в Томске.
Картинку рисует твой ComfyUI по твоему же рабочему процессу (workflow).

Настройка (один раз):
  1. В ComfyUI открой рабочий процесс, который у тебя рисует картинки (Z-Image-Turbo).
     В узле с положительной подсказкой (prompt) замени текст на {DREAM}.
  2. Меню → Экспорт (API) / Export (API) → сохрани как ~/stonkfly/dream_workflow.json
  3. ComfyUI должен быть доступен из Ubuntu: в настройках ComfyUI Desktop → Server-Config
     → Listen address = 0.0.0.0 (или задай свой адрес в COMFY_URL, см. ниже).
Переменные: COMFY_URL=http://IP:порт (по умолчанию ищет сам), DREAM_EVERY_MIN=60.
Результат: runs/dreams/latest.png, runs/dreams/dreams.json, runs/dreams/status.json
"""
import json
import os
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = Path("runs/dreams")
OUT.mkdir(parents=True, exist_ok=True)
EVERY = float(os.environ.get("DREAM_EVERY_MIN", 60)) * 60
WF = ROOT / "dream_workflow.json"
TOMSK = timezone(timedelta(hours=7))


def atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def status(ok, msg):
    atomic(OUT / "status.json", json.dumps({"ok": ok, "msg": msg, "wall": time.time()}, ensure_ascii=False).encode())
    print(("сон: " if ok else "сон, ошибка: ") + msg, flush=True)


# ---------- что происходит в мозге и на рынке ----------
REGIONS = [  # те же правила, что на 3D-странице
    ("vision", r"optic|visual|^ol|photo"), ("memory", None), ("central", r"central|^cb|intrinsic_brain|brain"),
    ("command", r"descend"), ("motor", r"motor|efferent"), ("senses", r"sensory|ascend"),
    ("cord", r"vnc|ventral|neck|ann|leg|wing"), ("other", r".*")]


def brain_mood():
    b = Path("runs/brain")
    try:
        meta = json.loads((b / "meta.json").read_text())
        cls = np.fromfile(b / "cls.bin", np.uint8)
        flag = np.fromfile(b / "flag.bin", np.uint8)
        act = np.fromfile(b / "act.bin", np.uint8)
    except Exception:
        return {}
    cmap = []
    for name in meta.get("classes", []):
        cmap.append(next(k for k, (_, rx) in enumerate(REGIONS) if rx and re.search(rx, name or "", re.I)))
    reg = np.array([cmap[c] if c < len(cmap) else len(REGIONS) - 1 for c in cls])
    reg[(flag & 1) > 0] = 1
    out = {}
    for k, (name, _) in enumerate(REGIONS):
        m = reg == k
        if m.any():
            out[name] = float((act[m] > 0).mean())
    return out


def market_mood():
    rows = []
    try:
        for line in Path("runs/paper/history.jsonl").read_text().splitlines()[-70:]:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        return {}
    if len(rows) < 2:
        return {}
    mid = lambda r: (float(r["quote"]["bid"]) + float(r["quote"]["ask"])) / 2
    return {
        "btc": (mid(rows[-1]) - mid(rows[0])) / mid(rows[0]),
        "equity": float(rows[-1]["equity_usdc"]) - float(rows[0]["equity_usdc"]),
        "reward": sum(r["neural"].get("stimulus") == "reward" for r in rows),
        "aversive": sum(r["neural"].get("stimulus") == "aversive" for r in rows),
        "buys": sum(r["neural"].get("side") == "BUY" for r in rows),
        "sells": sum(r["neural"].get("side") == "SELL" for r in rows),
    }


PH = {
    "vision": ["an endless kaleidoscope of hexagonal compound-eye facets", "a world seen through thousands of tiny glowing lenses"],
    "memory": ["giant glowing mushroom-shaped memory structures", "a cathedral of luminous mushroom bodies storing memories"],
    "central": ["a labyrinth of glowing neural highways", "a vast luminous web of neurons"],
    "command": ["tiny wings commanding the wind", "a fly pilot steering through the sky"],
    "reward": ["golden honey light pouring from the sky", "rivers of sugar crystals and warm golden sunlight"],
    "aversive": ["stormy violet shadows and falling rain", "a cold purple storm over a dark swamp"],
    "calm": ["a quiet misty meadow", "soft floating dust in still air"],
    "up": ["green candlestick towers rising into the clouds", "a green mountain range shaped like a price chart going up"],
    "down": ["red candlesticks falling like autumn leaves", "a crimson waterfall of falling price bars"],
    "flat": ["a perfectly flat horizon made of glowing lines", "an endless calm ocean of numbers"],
}
RU = {"vision": "зрение", "memory": "память", "central": "центр мозга", "command": "команды телу",
      "reward": "награды", "aversive": "наказания", "calm": "спокойствие", "up": "рост BTC", "down": "падение BTC", "flat": "тихий рынок"}


def compose():
    now = datetime.now(TOMSK)
    br, mk = brain_mood(), market_mood()
    parts, why = [], []
    top = sorted(((v, k) for k, v in br.items() if k in ("vision", "memory", "central", "command")), reverse=True)[:2]
    for _, k in top:
        parts.append(random.choice(PH[k]))
        why.append(RU[k])
    if mk:
        feel = "reward" if mk["reward"] > mk["aversive"] + 2 else "aversive" if mk["aversive"] > mk["reward"] + 2 else "calm"
        trend = "up" if mk["btc"] > .002 else "down" if mk["btc"] < -.002 else "flat"
        parts += [random.choice(PH[feel]), random.choice(PH[trend])]
        why += [RU[feel], RU[trend]]
    season = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer"}.get(now.month, "autumn")
    daytime = "night" if now.hour < 6 or now.hour >= 21 else "dawn" if now.hour < 9 else "evening" if now.hour >= 18 else "day"
    prompt = (f"surreal dream of a tiny fruit fly, {', '.join(parts)}, {season} {daytime} in a Siberian city, "
              "dreamy, magical, highly detailed, soft volumetric light, cinematic")
    caption = "снится: " + (", ".join(why) if why else "что-то тихое")
    return prompt, caption


# ---------- ComfyUI ----------
def candidates():
    if os.environ.get("COMFY_URL"):
        return [os.environ["COMFY_URL"].rstrip("/")]
    hosts = ["127.0.0.1"]
    try:
        gw = subprocess.run(["ip", "route"], capture_output=True, text=True).stdout
        m = re.search(r"default via (\S+)", gw)
        if m:
            hosts.append(m.group(1))
    except Exception:
        pass
    return [f"http://{h}:{p}" for h in hosts for p in (8000, 8188)]


def http(url, data=None, timeout=15):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_comfy():
    for u in candidates():
        try:
            http(u + "/system_stats", timeout=3)
            return u
        except Exception:
            continue
    return None


def fill(workflow, prompt):
    wf = json.loads(json.dumps(workflow))
    placed = False
    for node in wf.values():
        for k, v in node.get("inputs", {}).items():
            if isinstance(v, str) and "{DREAM}" in v:
                node["inputs"][k] = v.replace("{DREAM}", prompt)
                placed = True
            if k in ("seed", "noise_seed") and isinstance(v, int):
                node["inputs"][k] = random.randint(0, 2**48)
    if not placed:   # без метки: берём текстовый узел, который идёт в positive у сэмплера
        for node in wf.values():
            pos = node.get("inputs", {}).get("positive")
            if isinstance(pos, list) and str(pos[0]) in wf and "text" in wf[str(pos[0])].get("inputs", {}):
                wf[str(pos[0])]["inputs"]["text"] = prompt
                placed = True
                break
    if not placed:
        raise RuntimeError("в dream_workflow.json не нашёл, куда вставить подсказку: поставь {DREAM} в текст промпта")
    return wf


def dream(base):
    prompt, caption = compose()
    wf = fill(json.loads(WF.read_text()), prompt)
    pid = json.loads(http(base + "/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())}))["prompt_id"]
    for _ in range(600):
        time.sleep(2)
        hist = json.loads(http(f"{base}/history/{pid}"))
        if pid in hist:
            for out in hist[pid].get("outputs", {}).values():
                for img in out.get("images", []):
                    q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
                    data = http(f"{base}/view?{q}", timeout=60)
                    name = datetime.now(TOMSK).strftime("dream-%Y%m%d-%H%M.png")
                    (OUT / name).write_bytes(data)
                    atomic(OUT / "latest.png", data)
                    lst = json.loads((OUT / "dreams.json").read_text()) if (OUT / "dreams.json").exists() else []
                    lst = ([{"file": name, "wall": time.time(), "caption": caption, "prompt": prompt}] + lst)[:100]
                    atomic(OUT / "dreams.json", json.dumps(lst, ensure_ascii=False, indent=1).encode())
                    status(True, f"{caption} → {name}")
                    return
            raise RuntimeError("ComfyUI закончил, но картинку не вернул (есть ли в процессе узел Save Image?)")
    raise RuntimeError("ComfyUI не успел за 20 минут")


def main():
    print(f"Сны мухи: раз в {EVERY / 60:.0f} мин", flush=True)
    time.sleep(60)
    while True:
        if not WF.exists():
            status(False, "нет dream_workflow.json: экспортируй рабочий процесс из ComfyUI (см. начало dreams.py)")
            time.sleep(120)
            continue
        base = find_comfy()
        if not base:
            status(False, "ComfyUI не найден: " + ", ".join(candidates()) + ". Включи Listen 0.0.0.0 или задай COMFY_URL")
            time.sleep(120)
            continue
        try:
            dream(base)
        except Exception as e:
            status(False, str(e))
            time.sleep(300)
            continue
        time.sleep(EVERY)


if __name__ == "__main__":
    main()
