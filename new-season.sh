#!/usr/bin/env bash
# Архивирует текущий прогон и начинает новый сезон со свежими $100.
cd "$(dirname "$0")"
./stop.sh
n=$(cat runs/season 2>/dev/null || echo 1)
# итог сезона в runs/seasons.json (для статистики на экране)
[ -f runs/paper/events.jsonl ] && python3 - "$n" <<'P'
import json, sys, pathlib
n = int(sys.argv[1])
rows = [json.loads(l) for l in open("runs/paper/events.jsonl") if l.strip()]
if rows:
    mid = lambda r: (float(r["quote"]["bid"]) + float(r["quote"]["ask"])) / 2
    eq = [float(r["equity_usdc"]) for r in rows]
    e0 = eq[0]
    hodl = e0 * mid(rows[-1]) / mid(rows[0])
    peak, dd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e); dd = max(dd, (peak - e) / peak)
    s = {"season": n, "start": rows[0]["wall_time"], "end": rows[-1]["wall_time"], "ticks": len(rows),
         "start_equity": e0, "final_equity": eq[-1], "hodl_equity": hodl, "max_drawdown": dd,
         "trades": sum(1 for r in rows if r["execution"].get("status") == "FILLED")}
    p = pathlib.Path("runs/seasons.json")
    all_ = json.loads(p.read_text()) if p.exists() else []
    all_ = [x for x in all_ if x.get("season") != n] + [s]
    p.write_text(json.dumps(all_, indent=1))
    print(f"сезон {n}: {e0:.2f} -> {eq[-1]:.2f} USDC (HODL {hodl:.2f}), сделок {s['trades']}")
P
[ -d runs/paper ] && mv runs/paper "runs/season-$n-$(date +%F-%H%M)"
echo $((n+1)) > runs/season
echo "Сезон $((n+1))."
./start.sh
