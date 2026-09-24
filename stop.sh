#!/usr/bin/env bash
# Аккуратно останавливает муху (как Ctrl+C) и сервер.
tmux send-keys -t stonkfly:fly C-c 2>/dev/null; sleep 5
tmux kill-session -t stonkfly 2>/dev/null && echo "Остановлено." || echo "Не было запущено."
