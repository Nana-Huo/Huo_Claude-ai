@echo off

REM 添加Node.js和npm全局路径到环境变量，解决npm和ccr命令找不到的问题
set "PATH=%PATH%;C:\Program Files\nodejs;C:\Users\霍冠华\AppData\Roaming\npm"
"%ProgramFiles%\nodejs\node.exe" "%APPDATA%\npm\node_modules\zcf\bin\zcf.mjs" %*
