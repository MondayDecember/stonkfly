#!/usr/bin/env bash
# Запускает муху и веб-страницу в фоне (tmux-сессия stonkfly). Повторный запуск безопасен.
cd "$(dirname "$0")"
if ! tmux has-session -t stonkfly 2>/dev/null; then
  tmux new-session -d -s stonkfly -n fly ./runner.sh
  # помощники перезапускаются сами, если упадут
  tmux new-window  -t stonkfly -n web   "while true; do python3 -m http.server 8080 --bind 0.0.0.0; echo 'сервер упал, перезапуск'; sleep 5; done"
  tmux new-window  -t stonkfly -n brain "while true; do .venv/bin/python brainmap.py; echo 'карта мозга упала, перезапуск'; sleep 10; done"
  echo "Муха запущена."
else
  echo "Уже работает."
fi
IP=$(hostname -I | awk '{print $1}')
cat <<T

  3D-сцена:        http://localhost:8080/stonkfly-3d.html
  для OBS/стрима:  http://localhost:8080/stonkfly-3d.html?stream=1
  если localhost не открывается — http://$IP:8080/stonkfly-3d.html?stream=1

  смотреть логи:   tmux attach -t stonkfly   (выйти, не останавливая: Ctrl+B, затем D)
  остановить:      stonkfly stop
T
