@echo off

set NODE_PATH="C:\Program Files\nodejs\node.exe"
set CCR_CLI="C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@musistudio\claude-code-router\dist\cli.js"

%NODE_PATH% --version
%NODE_PATH% %CCR_CLI% ui

pause