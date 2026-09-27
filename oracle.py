#!/usr/bin/env python3
"""Муха-оракул: вопрос «да или нет» показывается мозгу мухи как картинка с текстом.

Муха смотрит на вопрос 0,5 с своего времени (~6 с счёта). Ответ снимается с тех же нейронов
DNp20, что у торговой мухи: правый заметно сильнее её обычного = ДА, слабее = НЕТ,
посередине = НЕ УВЕРЕНА. «Обычное» — медиана разницы П−Л на нейтральном кадре и прошлых вопросах
(у мухи есть постоянный перекос вправо). Память оракула заморожена: он не учится.

Откуда вопросы:
  * страница http://<IP>:8081/  (форма, удобно с телефона)
  * чат YouTube: сообщения, начинающиеся с «!муха» или «!fly», если задан YT_VIDEO=<id трансляции>
    (нужен пакет pytchat: .venv/bin/pip install pytchat)
Стоп-слова: файл oracle_banned.txt (по слову в строке); такие вопросы молча отбрасываются.
Состояние для 3D-страницы: runs/oracle/state.json
"""
import json
import os
import queue
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = Path("runs/oracle")
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("ORACLE_PORT", 8081))
MAXLEN, MAXQ = 120, 30
Q = queue.Queue(MAXQ)
STATE = {"current": None, "answers": [], "queue": 0, "ready": False}
LOCK = threading.Lock()


def atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def publish():
    with LOCK:
        STATE["queue"] = Q.qsize()
        atomic(OUT / "state.json", json.dumps(STATE, ensure_ascii=False).encode())


def banned():
    f = ROOT / "oracle_banned.txt"
    return [w.strip().lower() for w in f.read_text().splitlines() if w.strip()] if f.exists() else []


def submit(text, who, src):
    text = re.sub(r"\s+", " ", str(text or "")).strip()[:MAXLEN]
    who = re.sub(r"\s+", " ", str(who or "зритель")).strip()[:32] or "зритель"
    if len(text) < 3:
        return False, "слишком короткий вопрос"
    low = text.lower()
    if any(w in low for w in banned()):
        return False, "вопрос отклонён"
    try:
        Q.put_nowait({"q": text, "who": who, "src": src, "ts": time.time()})
    except queue.Full:
        return False, "очередь полна, попробуй позже"
    publish()
    ahead = max(0, Q.qsize() - 1) + (1 if STATE["current"] else 0)
    return True, "вопрос принят, муха уже смотрит" if not ahead else f"вопрос принят, перед тобой ещё {ahead}"


def render(text):
    """Кадр 320x180: чёрный текст на белом, как листок перед глазами мухи."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (320, 180), (250, 250, 245))
    d = ImageDraw.Draw(im)
    font = None
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            font = ImageFont.truetype(p, 20)
            break
    font = font or ImageFont.load_default(size=20)
    words, lines, line = text.split(), [], ""
    for w in words:
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) > 300 and line:
            lines.append(line)
            line = w
        else:
            line = t
    lines.append(line)
    y = max(8, (180 - 26 * len(lines)) // 2)
    for ln in lines[:6]:
        d.text(((320 - d.textlength(ln, font=font)) / 2, y), ln, fill=(10, 10, 10), font=font)
        y += 26
    return np.asarray(im, dtype=np.uint8)


def brain_worker():
    from stonkfly.config import Settings
    from stonkfly.neural.controller import FlyController

    print("Оракул: загружаю мозг мухи…", flush=True)
    ctl = FlyController(Settings(learning=False, neural_ms=500, pulse_ms=200))
    blank = np.full((180, 320, 3), 250, np.uint8)
    diffs = []
    for i in range(6):                         # калибровка: обычная разница П−Л на пустом листке
        diffs.append(ctl.observe(blank, "none")["difference_hz"])
    print(f"Оракул готов, обычная разница П−Л {statistics.median(diffs):+.1f} Гц", flush=True)
    with LOCK:
        STATE["ready"] = True
    publish()
    while True:
        item = Q.get()
        with LOCK:
            STATE["current"] = item
        publish()
        n = ctl.observe(render(item["q"]), "none")
        d = n["difference_hz"]
        base = statistics.median(diffs[-40:])
        spread = max(1.0, statistics.pstdev(diffs[-40:]) * .5)
        ans = "ДА" if d > base + spread else "НЕТ" if d < base - spread else "НЕ УВЕРЕНА"
        diffs.append(d)
        rec = {**item, "ans": ans, "diff": round(d - base, 2), "left": round(n["left_hz"], 1), "right": round(n["right_hz"], 1),
               "spikes": n["total_spikes"], "answered": time.time()}
        with LOCK:
            STATE["current"] = None
            STATE["answers"] = ([rec] + STATE["answers"])[:30]
        publish()
        with (OUT / "answers.jsonl").open("a") as h:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"«{item['q']}» ({item['who']}) → {ans}  (П−Л {d:+.1f} Гц, норма {base:+.1f})", flush=True)


def youtube_worker(video):
    try:
        import pytchat
    except ImportError:
        print("YT_VIDEO задан, но нет pytchat: .venv/bin/pip install pytchat", flush=True)
        return
    while True:
        try:
            chat = pytchat.create(video_id=video)
            print("Оракул: читаю чат YouTube", flush=True)
            while chat.is_alive():
                for c in chat.get().sync_items():
                    m = re.match(r"^\s*!(муха|fly)\s+(.+)", c.message, re.I)
                    if m:
                        submit(m.group(2), c.author.name, "youtube")
                time.sleep(2)
        except Exception as e:
            print("чат YouTube:", e, flush=True)
        time.sleep(30)


def web():
    from flask import Flask, jsonify, request, send_file
    app = Flask(__name__)

    @app.after_request
    def cors(r):
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r

    @app.get("/")
    def index():
        return send_file(ROOT / "oracle.html")

    @app.get("/state")
    def state():
        with LOCK:
            return jsonify(STATE)

    @app.post("/ask")
    def ask():
        data = request.get_json(silent=True) or request.form
        ok, msg = submit(data.get("q"), data.get("who"), "web")
        return jsonify({"ok": ok, "msg": msg}), (200 if ok else 400)

    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    publish()
    threading.Thread(target=brain_worker, daemon=True).start()
    if os.environ.get("YT_VIDEO"):
        threading.Thread(target=youtube_worker, args=(os.environ["YT_VIDEO"],), daemon=True).start()
    web()
