@echo off

REM 设置Node.js路径
set "NODE_PATH=C:\Program Files\nodejs"

REM 设置iFlow CLI路径
set "IFLOW_PATH=C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@iflow-ai\iflow-cli"

REM 设置环境变量
set "PATH=%PATH%;%NODE_PATH%"

REM 启动iFlow CLI
echo 启动iFlow CLI...
%NODE_PATH%\node.exe "%IFLOW_PATH%\bundle\iflow.js" %*

pause