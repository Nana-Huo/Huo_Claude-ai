@echo off
chcp 65001 >nul
echo 正在启动iFlow CLI...

REM 使用正确的路径和引号处理
"C:\Program Files\nodejs\node.exe" "C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@iflow-ai\iflow-cli\bundle\cli.js" %*

pause