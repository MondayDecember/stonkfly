#!/usr/bin/env bash
# Stonkfly: установка одной командой (Ubuntu 24.04 / WSL2).
# bash <(curl -fsSL https://raw.githubusercontent.com/MondayDecember/stonkfly/main/install.sh)
set -euo pipefail
REPO="${STONKFLY_REPO:-https://github.com/MondayDecember/stonkfly.git}"
DIR="${STONKFLY_DIR:-$HOME/stonkfly}"
say(){ printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }

say "Системные пакеты"
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential curl tmux ntpdate software-properties-common >/dev/null
if ! command -v python3.11 >/dev/null; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
  sudo apt-get update -qq
fi
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev >/dev/null

say "Синхронизация часов без пароля (нужна мухе: котировки проверяются с точностью 0.5 с)"
echo "$USER ALL=(root) NOPASSWD: /usr/sbin/ntpdate" | sudo tee /etc/sudoers.d/stonkfly-ntpdate >/dev/null
sudo chmod 440 /etc/sudoers.d/stonkfly-ntpdate
sudo visudo -cf /etc/sudoers.d/stonkfly-ntpdate >/dev/null
sudo -n ntpdate -u pool.ntp.org || true

say "Код"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only || true; else git clone "$REPO" "$DIR"; fi
cd "$DIR"
chmod +x ./*.sh

say "Python-окружение"
[ -d .venv ] || python3.11 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -e '.[test]'

say "Коннектом мухи (MaleCNS v1.0)"
if .venv/bin/python -m stonkfly verify >/dev/null 2>&1; then
  echo "уже на месте"
elif [ -f "$HOME/doomfly/doomfly/outputs/doom/malecns_v1/graph.npz" ] && \
     .venv/bin/python -m stonkfly prepare --reuse-doomfly "$HOME/doomfly/doomfly"; then
  echo "взят из doomfly"
else
  rm -rf data; .venv/bin/python -m stonkfly prepare
fi

say "Проверка доступа к Coinbase"
code=$(curl -s -o /dev/null -w '%{http_code}' https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USDC || true)
[ "$code" = 200 ] && echo "OK" || echo "ВНИМАНИЕ: Coinbase ответил $code, нужен обход (podkop/VPN)"

say "Готово. Запуск: ~/stonkfly/start.sh   (или ярлык Stonkfly.bat на рабочем столе)"
WINUSER=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r' || true)
if [ -n "$WINUSER" ] && [ -d "/mnt/c/Users/$WINUSER/Desktop" ]; then
  cp Stonkfly.bat "/mnt/c/Users/$WINUSER/Desktop/Stonkfly.bat" && echo "Ярлык положен на рабочий стол Windows"
fi
