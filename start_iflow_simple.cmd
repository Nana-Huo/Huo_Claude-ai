@echo off

REM 简单可靠的iFlow启动脚本
REM 这个批处理脚本使用完整路径，可以在任何Windows终端运行

REM 定义node.exe和iFlow CLI的完整路径
set NODE_PATH="C:\Program Files\nodejs\node.exe"
set IFLOW_PATH="C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@iflow-ai\iflow-cli\bundle\iflow.js"

REM 检查node.exe是否存在
if not exist %NODE_PATH% (
    echo 错误：找不到node.exe，请检查Node.js是否正确安装。
    echo 预期路径：%NODE_PATH%
    pause
    exit /b 1
)

REM 检查iFlow CLI是否存在
if not exist %IFLOW_PATH% (
    echo 错误：找不到iFlow CLI，请检查iFlow是否正确安装。
    echo 预期路径：%IFLOW_PATH%
    pause
    exit /b 1
)

REM 启动iFlow
echo 正在启动iFlow...
%NODE_PATH% %IFLOW_PATH%
