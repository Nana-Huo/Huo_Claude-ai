# 简化的Vercel部署脚本

# 设置Vercel API参数
$vercelToken = "BQB3W4ypldCYQXRnjnkv10Mk"
$projectId = "prj_qvUXbOd4HylZO7jclRPvSwKnhL5g"
$teamId = "team_ufijCPXi85fcJcNoJuHfa6Y4"
$apiUrl = "https://api.vercel.com/v13/deployments"

# 准备请求头
$headers = @{
    "Authorization" = "Bearer $vercelToken"
    "Content-Type" = "application/json"
}

# 准备请求体
$body = @{
    name = "dance-booking-app"
    projectId = $projectId
    ref = "main"
    target = "production"
    teamId = $teamId
}

# 转换为JSON
$jsonBody = ConvertTo-Json -InputObject $body -Compress

Write-Host "=== Vercel部署 ==="
Write-Host "项目ID: $projectId"
Write-Host "团队ID: $teamId"
Write-Host "分支: main"
Write-Host "环境: production"
Write-Host ""

try {
    # 使用Invoke-WebRequest调用API
    $response = Invoke-WebRequest -Uri $apiUrl -Method Post -Headers $headers -Body $jsonBody
    
    # 显示响应
    Write-Host "部署请求成功!"
    Write-Host "响应状态: $($response.StatusCode) $($response.StatusDescription)"
    Write-Host ""
    Write-Host "响应内容:"
    Write-Host $response.Content
    
} catch {
    # 处理错误
    Write-Host "部署请求失败:"
    Write-Host "错误信息: $($_.Exception.Message)"
    Write-Host ""
    
    # 尝试获取详细错误
    if ($_.Exception.Response) {
        $errorStream = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $errorContent = $errorStream.ReadToEnd()
        Write-Host "详细错误: $errorContent"
    }
}

Write-Host ""
Write-Host "=== 部署完成 ==="
