#!/usr/bin/env bash
# Аккуратно останавливает муху (как Ctrl+C), дожидаясь конца тика, потом помощников.
if ! tmux has-session -t stonkfly 2>/dev/null; then echo "Не было запущено."; exit 0; fi
tmux send-keys -t stonkfly:fly C-c 2>/dev/null
# тик считается ~6 с, Ctrl+C сработает после него; ждём до 30 с
for i in $(seq 30); do pgrep -f "stonkfly run" >/dev/null || break; sleep 1; done
pgrep -f "stonkfly run" >/dev/null && echo "Муха не ответила за 30 с, останавливаю принудительно."
tmux kill-session -t stonkfly 2>/dev/null
echo "Остановлено."
