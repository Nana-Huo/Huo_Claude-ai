# 最终Vercel部署脚本 - 简化版

# 设置变量
$vercel_token = "BQB3W4ypldCYQXRnjnkv10Mk"
$project_id = "prj_qvUXbOd4HylZO7jclRPvSwKnhL5g"
$team_id = "team_ufijCPXi85fcJcNoJuHfa6Y4"
$api_url = "https://api.vercel.com/v13/deployments"

# 手动构建JSON（避免PowerShell哈希表问题）
$json_body = "{
  \"projectId\": \"$project_id\",
  \"ref\": \"main\",
  \"target\": \"production\",
  \"teamId\": \"$team_id\"
}"

# 显示部署信息
Write-Host "=== Vercel部署 ==="
Write-Host "项目ID: $project_id"
Write-Host "团队ID: $team_id"
Write-Host "分支: main"
Write-Host "环境: production"
Write-Host ""

# 执行部署请求
Write-Host "正在发送部署请求..."
$response = Invoke-WebRequest -Uri $api_url -Method POST -Headers @{Authorization="Bearer $vercel_token";"Content-Type"="application/json"} -Body $json_body

# 显示结果
Write-Host ""
Write-Host "部署请求响应:"
Write-Host "状态代码: $($response.StatusCode)"
Write-Host "状态描述: $($response.StatusDescription)"
Write-Host ""
Write-Host "响应内容:"
Write-Host $response.Content
