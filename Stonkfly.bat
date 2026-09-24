@echo off
chcp 65001 >nul
title Stonkfly
wsl -d Ubuntu-24.04 -- bash -lc "~/stonkfly/start.sh"
for /f "tokens=1" %%i in ('wsl -d Ubuntu-24.04 -- hostname -I') do set IP=%%i
start "" "http://%IP%:8080/stonkfly-3d.html?stream=1"
echo.
echo Это окно показывает логи мухи. Не закрывай его, можно свернуть.
wsl -d Ubuntu-24.04 -- bash -lc "tmux attach -t stonkfly"
