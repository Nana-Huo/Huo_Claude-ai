@echo off

REM Simple iFlow launcher
REM This script uses only English to avoid encoding issues

REM Define full paths
set NODE_PATH=C:\Program Files\nodejs\node.exe
set IFLOW_PATH=C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@iflow-ai\iflow-cli\bundle\iflow.js

REM Start iFlow
%NODE_PATH% %IFLOW_PATH%