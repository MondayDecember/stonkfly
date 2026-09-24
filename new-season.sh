#!/usr/bin/env bash
# Архивирует текущий прогон и начинает новый сезон со свежими $100.
cd "$(dirname "$0")"
./stop.sh
n=$(cat runs/season 2>/dev/null || echo 1)
[ -d runs/paper ] && mv runs/paper "runs/season-$n-$(date +%F-%H%M)"
echo $((n+1)) > runs/season
echo "Сезон $((n+1))."
./start.sh
