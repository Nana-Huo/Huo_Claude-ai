@echo off
setlocal enabledelayedexpansion

REM Vercel API部署脚本
set "VERCEL_TOKEN=BQB3W4ypldCYQXRnjnkv10Mk"
set "PROJECT_ID=prj_qvUXbOd4HylZO7jclRPvSwKnhL5g"
set "TEAM_ID=team_ufijCPXi85fcJcNoJuHfa6Y4"
set "API_ENDPOINT=https://api.vercel.com/v13/deployments"

REM 创建JSON请求体
set "JSON_BODY={"name":"dance-booking-app","projectId":"%PROJECT_ID%","ref":"main","target":"production","teamId":"%TEAM_ID%"}"

REM 替换双引号以确保JSON格式正确
set "JSON_BODY=!JSON_BODY:^"="!"

echo === 开始Vercel部署 ===
echo 项目ID: %PROJECT_ID%
echo 团队ID: %TEAM_ID%
echo 分支: main
echo 环境: production
echo.

echo 正在调用Vercel API...
echo.

REM 使用curl调用Vercel API
curl -X POST "%API_ENDPOINT%" ^
  -H "Authorization: Bearer %VERCEL_TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d "%JSON_BODY%"

echo.
echo === 部署完成 ===
