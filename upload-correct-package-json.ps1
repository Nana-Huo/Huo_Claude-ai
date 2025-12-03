# 配置信息
$GITHUB_TOKEN = "github_pat_11AQX4B5A06BdY59b7Fb1bR58N3t3X5X9HxXaV4Bd3bR58N3t3X5X9HxXaV4Bd3"
$OWNER = "Nana-Huo"
$REPO = "dance-booking-app"
$BRANCH = "master"
$FILE_PATH = "package.json"

# 本地文件路径
$LOCAL_FILE_PATH = "C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app\package.json"

Write-Host "Uploading correct package.json to GitHub..." -ForegroundColor Green

# 设置请求头
$headers = @{
    "Authorization" = "token $GITHUB_TOKEN"
    "Content-Type" = "application/json"
    "Accept" = "application/vnd.github.v3+json"
}

# 1. 获取最新的提交SHA
Write-Host "\n1. Getting latest commit SHA..." -ForegroundColor Yellow
$branchUrl = "https://api.github.com/repos/$OWNER/$REPO/git/ref/heads/$BRANCH"
$branchData = Invoke-RestMethod -Uri $branchUrl -Headers $headers -Method Get
$latestCommitSha = $branchData.object.sha
Write-Host "✅ Latest commit SHA: $latestCommitSha" -ForegroundColor Green

# 2. 读取本地文件内容并转换为Base64
Write-Host "\n2. Reading local package.json..." -ForegroundColor Yellow
$localContent = Get-Content -Path $LOCAL_FILE_PATH -Raw -Encoding UTF8
$base64Content = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($localContent))
Write-Host "✅ Local file read successfully" -ForegroundColor Green

# 3. 创建新文件的Blob
Write-Host "\n3. Creating blob for package.json..." -ForegroundColor Yellow
$blobUrl = "https://api.github.com/repos/$OWNER/$REPO/git/blobs"
$blobData = @{
    "content" = $base64Content
    "encoding" = "base64"
} | ConvertTo-Json

$blobResponse = Invoke-RestMethod -Uri $blobUrl -Headers $headers -Method Post -Body $blobData
$newBlobSha = $blobResponse.sha
Write-Host "✅ New blob SHA: $newBlobSha" -ForegroundColor Green

# 4. 获取当前树结构
Write-Host "\n4. Getting current tree structure..." -ForegroundColor Yellow
$commitUrl = "https://api.github.com/repos/$OWNER/$REPO/git/commits/$latestCommitSha"
$commitData = Invoke-RestMethod -Uri $commitUrl -Headers $headers -Method Get
$currentTreeSha = $commitData.tree.sha
Write-Host "✅ Current tree SHA: $currentTreeSha" -ForegroundColor Green

# 5. 创建新的树结构，替换package.json
Write-Host "\n5. Creating new tree structure..." -ForegroundColor Yellow
$newTreeData = @{
    "base_tree" = $currentTreeSha
    "tree" = @(
        @{
            "path" = $FILE_PATH
            "mode" = "100644"
            "type" = "blob"
            "sha" = $newBlobSha
        }
    )
} | ConvertTo-Json

$newTreeUrl = "https://api.github.com/repos/$OWNER/$REPO/git/trees"
$newTreeResponse = Invoke-RestMethod -Uri $newTreeUrl -Headers $headers -Method Post -Body $newTreeData
$newTreeSha = $newTreeResponse.sha
Write-Host "✅ New tree SHA: $newTreeSha" -ForegroundColor Green

# 6. 创建新的提交
Write-Host "\n6. Creating new commit..." -ForegroundColor Yellow
$newCommitData = @{
    "message" = "Fix broken package.json and restore correct JSON format"
    "tree" = $newTreeSha
    "parents" = @($latestCommitSha)
} | ConvertTo-Json

$newCommitUrl = "https://api.github.com/repos/$OWNER/$REPO/git/commits"
$newCommitResponse = Invoke-RestMethod -Uri $newCommitUrl -Headers $headers -Method Post -Body $newCommitData
$newCommitSha = $newCommitResponse.sha
Write-Host "✅ New commit SHA: $newCommitSha" -ForegroundColor Green

# 7. 更新分支引用
Write-Host "\n7. Updating branch reference..." -ForegroundColor Yellow
$updateRefUrl = "https://api.github.com/repos/$OWNER/$REPO/git/refs/heads/$BRANCH"
$updateRefData = @{
    "sha" = $newCommitSha
    "force" = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri $updateRefUrl -Headers $headers -Method Patch -Body $updateRefData
Write-Host "✅ Branch reference updated" -ForegroundColor Green

Write-Host "\n🎉 Package.json has been successfully fixed and uploaded to GitHub!" -ForegroundColor Green
Write-Host "You can now try redeploying on Vercel." -ForegroundColor Cyan