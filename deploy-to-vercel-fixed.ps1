# Vercel API部署脚本
param(
    [string]$VercelToken,
    [string]$ProjectId,
    [string]$TeamId = ""
)

# 检查参数
if (-not $VercelToken) {
    Write-Host "错误: 必须提供Vercel API Token" -ForegroundColor Red
    Write-Host "使用方法: .\deploy-to-vercel-fixed.ps1 -VercelToken "your-token" -ProjectId "your-project-id" [-TeamId "your-team-id"]" -ForegroundColor White
    exit 1
}

if (-not $ProjectId) {
    Write-Host "错误: 必须提供Vercel Project ID" -ForegroundColor Red
    Write-Host "使用方法: .\deploy-to-vercel-fixed.ps1 -VercelToken "your-token" -ProjectId "your-project-id" [-TeamId "your-team-id"]" -ForegroundColor White
    exit 1
}

# 设置API端点
$apiEndpoint = "https://api.vercel.com/v13/deployments"

# 设置请求头
$headers = @{
    "Authorization" = "Bearer $VercelToken"
    "Content-Type" = "application/json"
}

# 设置请求体
$body = @{
    name = "dance-booking-app"
    projectId = $ProjectId
    ref = "main"
    target = "production"
}

# 如果有Team ID，添加到请求体
if ($TeamId) {
    $body.Add("teamId", $TeamId)
}

# 转换为JSON
$jsonBody = $body | ConvertTo-Json

Write-Host "=== 开始Vercel部署 ===" -ForegroundColor Green
Write-Host "项目ID: $ProjectId" -ForegroundColor Yellow
if ($TeamId) {
    Write-Host "团队ID: $TeamId" -ForegroundColor Yellow
}
Write-Host "分支: main"
Write-Host "环境: production"

try {
    # 调用Vercel API
    $response = Invoke-RestMethod -Uri $apiEndpoint -Method Post -Headers $headers -Body $jsonBody -UseBasicParsing
    
    Write-Host "\n=== 部署请求成功 ===" -ForegroundColor Green
    Write-Host "部署ID: $($response.id)" -ForegroundColor Yellow
    Write-Host "部署URL: $($response.url)" -ForegroundColor Cyan
    Write-Host "状态: $($response.status)" -ForegroundColor White
    Write-Host "创建时间: $($response.createdAt)" -ForegroundColor White
    
    Write-Host "\n=== 部署状态查询 ===" -ForegroundColor Yellow
    Write-Host "要查看部署进度，可以访问: $($response.url)" -ForegroundColor White
    Write-Host "或使用部署ID查询状态: https://api.vercel.com/v13/deployments/$($response.id)" -ForegroundColor White
    
    return $response
} catch {
    Write-Host "\n=== 部署请求失败 ===" -ForegroundColor Red
    Write-Host "错误信息: $($_.Exception.Message)" -ForegroundColor White
    
    # 输出详细错误
    if ($_.Exception.Response) {
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $errorContent = $reader.ReadToEnd()
            Write-Host "详细错误: $errorContent" -ForegroundColor Red
        } catch {
            Write-Host "无法读取详细错误内容" -ForegroundColor White
        }
    }
    
    exit 1
}
