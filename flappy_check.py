#!/usr/bin/env python3
"""Учится ли муха играть во Flappy Fly? Три честные проверки.

1. Счёт: сравнивает первые и последние попытки между собой и с мухой, машущей наугад
   с той же частотой (бутстреп, 95% интервал).
2. Мозг: сколько синапсов памяти изменилось и насколько, и как далеко по коннектому
   место обучения (MBON07/MBON11) от нейронов, которые решают взмах (DNp20, DNpe017).
3. Опыт с отключением памяти: одни и те же кадры игры показываются мозгу дважды,
   с выученными синапсами и с исходными. Всё остальное одинаково (симуляция детерминирована),
   поэтому любая разница в решениях — это вклад обучения.

Запуск (игру можно не останавливать, просто будет медленнее):
  cd ~/stonkfly && .venv/bin/python flappy_check.py            # всё, ~5–10 минут
  .venv/bin/python flappy_check.py --quick                     # только счёт, секунды
  .venv/bin/python flappy_check.py --frames 200                # опыт на 200 кадрах
"""
import argparse
import json
import sys
import time

import numpy as np

import flappyfly as ff   # те же правила игры и та же планка «наугад»; меняет папку на корень проекта


def say(*a):
    print(*a, flush=True)


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    return np.array([rng.choice(x, len(x)).mean() for _ in range(n)])


def check_scores():
    say("\n=== 1. СЧЁТ ПО ПОПЫТКАМ ===")
    path = ff.OUT / "attempts.jsonl"
    if not path.exists():
        say("попыток ещё нет")
        return None
    a = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(a) < 40:
        say(f"попыток {len(a)}, для выводов нужно хотя бы 40")
        return None
    sc = np.array([x["score"] for x in a], float)
    st = np.array([x["steps"] for x in a], float)
    fl = np.array([x["flaps"] for x in a], float)
    say(f"попыток: {len(a)}, правила: {ff.LEVEL}")
    B = max(20, len(a) // 8)
    say(f"\nпо блокам из {B} попыток:  средний счёт | прожито шагов | доля взмахов | наугад при той же доле")
    table = {}
    for i in range(0, len(a) - B + 1, B):
        s = slice(i, i + B)
        rate = fl[s].sum() / st[s].sum()
        r = round(rate, 2)
        if r not in table:
            table[r] = ff.random_baseline(rate, n=300)
        say(f"  {i + 1:>5}–{i + B:<5}  {sc[s].mean():6.2f}         {st[s].mean():7.1f}        {rate:5.2f}          {table[r]:5.2f}")
    k = min(100, len(a) // 3)
    first, last = sc[:k], sc[-k:]
    d = boot(last, seed=1) - boot(first, seed=2)
    lo, hi = np.percentile(d, [2.5, 97.5])
    rate = fl[-k:].sum() / st[-k:].sum()
    base = ff.random_baseline(rate, n=600)
    bl = boot(last, seed=3) - base
    blo, bhi = np.percentile(bl, [2.5, 97.5])
    say(f"\nпервые {k}: {first.mean():.2f} трубы за попытку, последние {k}: {last.mean():.2f}")
    say(f"рост: {last.mean() - first.mean():+.2f} (95% интервал {lo:+.2f} … {hi:+.2f})")
    say(f"последние {k} против мухи, машущей наугад ({base:.2f}): {last.mean() - base:+.2f} (95% интервал {blo:+.2f} … {bhi:+.2f})")
    if lo > 0:
        say("→ счёт заметно вырос: похоже на обучение")
    elif hi < 0:
        say("→ счёт заметно упал")
    else:
        say("→ разница в пределах случайности: по счёту обучения не видно")
    if blo > 0:
        say("→ муха играет лучше, чем наугад")
    elif bhi < 0:
        say("→ муха играет хуже, чем наугад")
    else:
        say("→ не отличается от игры наугад")
    return {"attempts": len(a), "first": first.mean(), "last": last.mean(), "ci": (lo, hi), "vs_random": (blo, bhi)}


def edge_chunks(brain, size=4_000_000):
    """Синапсы порциями, чтобы не держать в памяти лишние массивы на весь коннектом."""
    E = len(brain.post)
    for s in range(0, E, size):
        e = min(E, s + size)
        pre = (np.searchsorted(brain.ptr, np.arange(s, e), side="right") - 1).astype(np.int32)
        yield s, e, pre


def hops(brain, sources, targets, limit=8):
    """Минимальное число синапсов от любого из sources до любого из targets."""
    seen = np.zeros(brain.n, bool)
    seen[sources] = True
    front = seen.copy()
    for h in range(1, limit + 1):
        nxt = np.zeros(brain.n, bool)
        for s, e, pre in edge_chunks(brain):
            m = front[pre]
            nxt[brain.post[s:e][m]] = True
        nxt &= ~seen
        if nxt[targets].any():
            return h
        if not nxt.any():
            return None
        seen |= nxt
        front = nxt
    return None


def input_share(brain, sources, targets, steps=4):
    """Доля входа targets, которая приходит от sources по путям длиной 1..steps.
    Каждый нейрон делит свой вход по долям |веса| входящих синапсов (как «effective input» в коннектомике)."""
    total = np.zeros(brain.n)
    for s, e, pre in edge_chunks(brain):
        total += np.bincount(brain.post[s:e], weights=np.abs(brain.weight[s:e]), minlength=brain.n)
    total = np.maximum(total, 1e-9)
    x = np.zeros(brain.n)
    x[sources] = 1
    out = []
    for _ in range(steps):
        y = np.zeros(brain.n)
        for s, e, pre in edge_chunks(brain):
            post = brain.post[s:e]
            y += np.bincount(post, weights=x[pre] * np.abs(brain.weight[s:e]) / total[post], minlength=brain.n)
        y[sources] = 0
        x = y
        out.append(float(x[targets].mean()))
    return out


def check_brain(ctl):
    say("\n=== 2. МОЗГ И ПАМЯТЬ ===")
    b = ctl.brain
    e = b.circuit["edges"]
    ratio = b.weight[e] / b.baseline_plastic
    ch = np.abs(ratio - 1) > 1e-6
    say(f"синапсов памяти (KC→MBON07/MBON11): {len(e)}, изменились: {int(ch.sum())} ({ch.mean() * 100:.0f}%)")
    if ch.any():
        say(f"изменение силы: среднее {np.abs(ratio[ch] - 1).mean() * 100:.1f}%, самое большое {np.abs(ratio - 1).max() * 100:.1f}%, "
            f"ослаблены {int((ratio < 1 - 1e-6).sum())}, усилены {int((ratio > 1 + 1e-6).sum())}")
    mb, dec = b.circuit["mb"], ctl.decoder
    names = {"DNp20 левый": dec.left, "DNp20 правый": dec.right, "DNpe017": dec.gate}
    say("\nкак далеко выход памяти (MBON07, MBON11) от нейронов, которые решают взмах:")
    for name, ix in names.items():
        h = hops(b, mb, ix)
        say(f"  {name}: {h if h else 'нет пути'} синапс(а) по кратчайшему пути")
    say("доля входа нейронов решения, приходящая от MBON07/MBON11 (по путям длиной 1, 2, 3, 4 синапса):")
    for name, ix in names.items():
        sh = input_share(b, mb, ix)
        say(f"  {name}: " + ", ".join(f"{v * 100:.3f}%" for v in sh))


def frames(n, seed=7):
    """Один и тот же набор кадров для обоих мозгов: игра, в которой машут наугад (20%)."""
    rng = np.random.default_rng(seed)
    g = ff.Game(rng)
    out = []
    while len(out) < n:
        out.append(g.frame())
        _, crashed = g.step(rng.random() < ff.FLAP_RATE, ff.STEP_MS / 1000)
        if crashed:
            g.reset()
    return out


def run(ctl, fr, w):
    b = ctl.brain
    b.reset(keep_memory=True)
    b.weight[b.circuit["edges"]] = w
    sc, df, kc = [], [], []
    for i, f in enumerate(fr):
        n = ctl.observe(f, "none")
        sc.append(ff.decision_score(n))
        df.append(n["difference_hz"])
        kc.append(n.get("KC_spikes", 0))
        if (i + 1) % 25 == 0:
            say(f"    кадр {i + 1}/{len(fr)}")
    return np.array(sc), np.array(df), np.array(kc, float)


def flaps(sc):
    """То же правило, что в игре: взмах, когда оценка в верхних 20% (по этому прогону)."""
    thr = np.percentile(sc, 100 * (1 - ff.FLAP_RATE))
    return sc >= thr


def check_ablation(ctl, n):
    say(f"\n=== 3. ОПЫТ: ВЫУЧЕННЫЕ СИНАПСЫ ПРОТИВ ИСХОДНЫХ ({n} кадров) ===")
    b = ctl.brain
    learned = b.weight[b.circuit["edges"]].copy()
    if np.allclose(learned, b.baseline_plastic):
        say("память ещё не изменилась, сравнивать не с чем")
        return
    fr = frames(n)
    t = time.time()
    say("  прогон с выученной памятью…")
    s1, d1, k1 = run(ctl, fr, learned)
    say("  прогон с исходной памятью…")
    s0, d0, k0 = run(ctl, fr, b.baseline_plastic.copy())
    b.weight[b.circuit["edges"]] = learned
    f1, f0 = flaps(s1), flaps(s0)
    same = (s1 == s0).mean()
    diff = np.abs(d1 - d0)
    say(f"  готово за {time.time() - t:.0f} с")
    say(f"кадров, где выход мозга в точности совпал: {same * 100:.0f}%")
    say(f"разница П−Л DNp20: в среднем {diff.mean():.2f} Гц (разброс самого сигнала от кадра к кадру {d0.std():.2f} Гц)")
    say(f"решение «махать / не махать» разное в {(f1 != f0).mean() * 100:.0f}% кадров")
    if k0.mean() > 0:
        say(f"клетки Кеньона (вход памяти): {k0.mean():.0f} спайков за кадр, от кадра к кадру меняется на {k0.std() / k0.mean() * 100:.0f}%")
    if (f1 != f0).mean() < .02:
        say("→ выученная память на решения почти не влияет: учиться играть мухе нечем")
    else:
        say("→ память меняет решения; учится ли она правильным решениям, покажет счёт (проверка 1)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="только счёт")
    p.add_argument("--frames", type=int, default=120)
    args = p.parse_args()
    check_scores()
    if args.quick:
        return
    ckpt = ff.OUT / "brain.npz"
    if not ckpt.exists():
        say("\nнет runs/flappy/brain.npz: память сохраняется каждые 10 попыток, подожди")
        return
    from stonkfly.config import Settings
    from stonkfly.neural.controller import FlyController
    say("\nзагружаю мозг мухи (как в игре, но без обучения)…")
    ctl = FlyController(Settings(learning=False, neural_ms=ff.STEP_MS, pulse_ms=min(200, ff.STEP_MS)))
    ctl.restore(ckpt)
    ctl.brain.weights_frozen = True
    check_brain(ctl)
    check_ablation(ctl, args.frames)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
