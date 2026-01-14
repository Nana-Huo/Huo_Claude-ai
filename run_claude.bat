@echo off
:: 简单的Claude启动脚本
:: 确保Node.js和npm在PATH中
set NODE_PATH=C:\Program Files\nodejs
set NPM_PATH=C:\Users\霍冠华\AppData\Roaming\npm
set PATH=%NODE_PATH%;%NPM_PATH%;%PATH%

:: 启动Claude
claude %*