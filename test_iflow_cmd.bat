@echo off
chcp 65001 >nul
echo Testing iflow in CMD...
echo.
iflow --version
echo.
echo Exit code: %errorlevel%
pause