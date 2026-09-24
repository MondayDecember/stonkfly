@echo off
chcp 65001 >nul
title Stonkfly: установка
echo === Stonkfly: установка ===
echo.
wsl -d Ubuntu-24.04 -- true >nul 2>&1
if errorlevel 1 (
  echo Ubuntu 24.04 для WSL не найдена, ставлю. Windows может попросить права администратора.
  wsl --install -d Ubuntu-24.04
  echo.
  echo Когда Ubuntu попросит придумать имя пользователя и пароль, придумай их.
  echo Потом запусти этот файл ещё раз. Если Windows попросит перезагрузку, перезагрузись.
  pause
  exit /b
)
echo Ubuntu найдена. Ставлю муху (понадобится пароль от Ubuntu)...
echo.
wsl -d Ubuntu-24.04 -- bash -lc "bash <(curl -fsSL https://raw.githubusercontent.com/MondayDecember/stonkfly/main/install.sh)"
echo.
pause
