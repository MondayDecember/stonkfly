@echo off
chcp 65001 >nul
wsl -d Ubuntu-24.04 -- bash -lc "~/stonkfly/stonkfly-cli open"
start "Stonkfly: логи (не закрывай, можно свернуть)" /min wsl -d Ubuntu-24.04 -- bash -lc "tmux attach -t stonkfly"
