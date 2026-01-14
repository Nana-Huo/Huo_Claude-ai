@echo off

:: Add Node.js and npm to PATH
set PATH=C:\Program Files\nodejs;C:\Users\霍冠华\AppData\Roaming\npm;%PATH%

:: Run Claude
node "C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\cli.js" %*