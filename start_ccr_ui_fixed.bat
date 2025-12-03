@echo off

REM 保存原始环境变量
set ORIGINAL_PATH=%PATH%

REM 设置完整的环境变量
set "NODE_PATH=C:\Program Files\nodejs"
set "NPM_GLOBAL=C:\Users\霍冠华\AppData\Roaming\npm"
set "PATH=%NODE_PATH%;%NPM_GLOBAL%;%PATH%"

REM 设置NODE_EXEC_PATH环境变量，让CCR知道node的位置
set "NODE_EXEC_PATH=%NODE_PATH%\node.exe"

REM 检查node是否可用
echo 检查Node.js版本:
"%NODE_PATH%\node.exe" --version
if %ERRORLEVEL% NEQ 0 (
    echo Node.js未找到，请检查路径是否正确
    pause
    exit /b 1
)

REM 使用完整路径执行CCR UI
echo 启动CCR UI...
"%NODE_PATH%\node.exe" "%NPM_GLOBAL%\node_modules\@musistudio\claude-code-router\dist\cli.js" ui

REM 恢复原始环境变量
set PATH=%ORIGINAL_PATH%

pause