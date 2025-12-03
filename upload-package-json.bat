@echo off
setlocal enabledelayedexpansion

rem 设置配置信息
set GITHUB_TOKEN=github_pat_11AQX4B5A06BdY59b7Fb1bR58N3t3X5X9HxXaV4Bd3bR58N3t3X5X9HxXaV4Bd3
set OWNER=Nana-Huo
set REPO=dance-booking-app
set BRANCH=master
set FILE_PATH=package.json
set LOCAL_FILE=C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app\package.json

rem 读取本地文件内容
set /p CONTENT=<%LOCAL_FILE%

rem 创建临时文件用于存储JSON数据
echo {
  echo   "message": "Fix package.json JSON format error",
  echo   "content": "" > temp.json

rem 使用PowerShell进行Base64编码
powershell -Command "$content = Get-Content -Path '%LOCAL_FILE%' -Raw; $base64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($content)); Add-Content -Path 'temp.json' -Value $base64" 

echo   ,
  echo   "branch": "master"
  echo } >> temp.json

rem 使用curl上传文件
curl -X PUT "https://api.github.com/repos/%OWNER%/%REPO%/contents/%FILE_PATH%" ^
  -H "Authorization: token %GITHUB_TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d @temp.json

rem 清理临时文件
del temp.json

echo.
echo Upload completed!