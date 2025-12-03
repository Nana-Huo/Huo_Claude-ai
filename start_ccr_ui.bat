@echo off

REM 设置Node.js和npm全局路径到环境变量
set "NODE_PATH=C:\Program Files\nodejs"
set "NPM_GLOBAL=C:\Users\霍冠华\AppData\Roaming\npm"
set "PATH=%PATH%;%NODE_PATH%;%NPM_GLOBAL%"

REM 查找CCR的实际安装路径
for /f "delims=" %%i in ('%NODE_PATH%\npm.cmd list -g @musistudio/claude-code-router --silent') do set "CCR_INSTALL=%%i"

REM 提取CCR的实际路径（去掉版本号）
for %%i in ("%CCR_INSTALL%") do set "CCR_PATH=%%~dpi"

REM 使用完整路径执行CCR UI
%NODE_PATH%\node.exe "%CCR_PATH%.bin\ccr.js" ui

REM 等待用户输入
pause