#!/usr/bin/env bash
# Держит муху запущенной: синхронизирует часы и перезапускает после временных остановок.
cd "$(dirname "$0")"
source .venv/bin/activate
RUN=runs/paper
mkdir -p runs; [ -f runs/season ] || echo 1 > runs/season
status(){ printf '{"state":"%s","season":%s,"reason":"%s","since":%s}\n' \
  "$1" "$(cat runs/season)" "${2//\"/}" "$(date +%s)" > runs/status.json; }
reason(){ python3 -c "import json;print(json.load(open('$RUN/error.json'))['reason'])" 2>/dev/null; }

while true; do
  r=$(reason)
  if [[ "$r" == *"Loss stop"* ]]; then
    status gameover "$r"
    echo "GAME OVER: муха проиграла лимит. Новый сезон: ~/stonkfly/new-season.sh"
    exec sleep infinity
  fi
  sudo -n ntpdate -u pool.ntp.org >/dev/null 2>&1
  rm -f "$RUN/error.json"
  status running ""
  if [ -f "$RUN/ledger.sqlite" ]; then python -m stonkfly run --resume-reviewed; else python -m stonkfly run; fi
  code=$?
  [ $code -eq 0 ] && { status stopped ""; break; }
  r=$(reason)
  [[ "$r" == *"Loss stop"* ]] && continue
  echo "Остановка: ${r:-неизвестно}. Перезапуск через 20 с…"
  status restarting "$r"
  sleep 20
done
