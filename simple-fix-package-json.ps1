# 简化版修复脚本
$GITHUB_TOKEN = "github_pat_11AQX4B5A06BdY59b7Fb1bR58N3t3X5X9HxXaV4Bd3bR58N3t3X5X9HxXaV4Bd3"
$OWNER = "Nana-Huo"
$REPO = "dance-booking-app"
$BRANCH = "main"

# 本地package.json路径
$localPath = "C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app\package.json"

# 读取本地正确的package.json内容
$packageJsonContent = Get-Content -Path $localPath -Raw -Encoding UTF8

# 将内容转换为Base64
$base64Content = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($packageJsonContent))

# 设置API请求头
$headers = @{
    "Authorization" = "token $GITHUB_TOKEN"
    "Content-Type" = "application/json"
}

# 1. 获取当前文件的SHA
$fileInfoUrl = "https://api.github.com/repos/$OWNER/$REPO/contents/package.json?ref=$BRANCH"
$fileInfo = Invoke-RestMethod -Uri $fileInfoUrl -Headers $headers -Method Get
$currentSha = $fileInfo.sha

# 2. 更新文件
$updateUrl = "https://api.github.com/repos/$OWNER/$REPO/contents/package.json"
$updateData = @{
    message = "Fix package.json syntax error"
    content = $base64Content
    sha = $currentSha
    branch = $BRANCH
} | ConvertTo-Json

$updateResult = Invoke-RestMethod -Uri $updateUrl -Headers $headers -Method Put -Body $updateData

Write-Host "Package.json updated successfully!"
Write-Host "Commit SHA: $($updateResult.commit.sha)"