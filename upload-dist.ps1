# GitHub API 配置
$GITHUB_TOKEN = "github_pat_11AQX4B5A06BdY59b7Fb1bR58N3t3X5X9HxXaV4Bd3bR58N3t3X5X9HxXaV4Bd3"
$OWNER = "Nana-Huo"
$REPO = "dance-booking-app"
$BRANCH = "main"

# 项目路径
$PROJECT_ROOT = "C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app"
$SERVER_DIST_PATH = "$PROJECT_ROOT\server\dist"

Write-Host "开始上传 server/dist 目录到 GitHub 仓库..." -ForegroundColor Green

# 设置 API 头部
$headers = @{
    "Authorization" = "token $GITHUB_TOKEN"
    "Accept" = "application/vnd.github.v3+json"
}

# 1. 获取当前分支的最新提交 SHA
Write-Host "\n1. 获取当前分支 $BRANCH 的最新提交..." -ForegroundColor Yellow
$branchUrl = "https://api.github.com/repos/$OWNER/$REPO/branches/$BRANCH"
$branchData = Invoke-RestMethod -Uri $branchUrl -Headers $headers -Method Get
$latestCommitSha = $branchData.commit.sha
Write-Host "✅ 最新提交 SHA: $latestCommitSha" -ForegroundColor Green

# 2. 获取当前树的 SHA
$commitUrl = "https://api.github.com/repos/$OWNER/$REPO/git/commits/$latestCommitSha"
$commitData = Invoke-RestMethod -Uri $commitUrl -Headers $headers -Method Get
$currentTreeSha = $commitData.tree.sha
Write-Host "✅ 当前树 SHA: $currentTreeSha" -ForegroundColor Green

# 3. 收集 dist 目录下的所有文件
Write-Host "\n2. 收集 server/dist 目录下的文件..." -ForegroundColor Yellow
$files = Get-ChildItem -Path $SERVER_DIST_PATH -Recurse -File
Write-Host "✅ 找到 $($files.Count) 个文件" -ForegroundColor Green

# 4. 为每个文件创建 blob 并构建树对象
Write-Host "\n3. 为文件创建 blob 并构建树对象..." -ForegroundColor Yellow
$treeItems = @()

foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($PROJECT_ROOT.Length + 1)
    Write-Host "处理文件: $relativePath" -ForegroundColor Cyan
    
    # 读取文件内容
    $content = Get-Content -Path $file.FullName -Raw
    $base64Content = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($content))
    
    # 创建 blob
    $blobUrl = "https://api.github.com/repos/$OWNER/$REPO/git/blobs"
    $blobData = @{
        content = $base64Content
        encoding = "base64"
    }
    $blobResult = Invoke-RestMethod -Uri $blobUrl -Headers $headers -Method Post -Body ($blobData | ConvertTo-Json)
    
    # 添加到树对象
    $treeItem = @{
        path = $relativePath
        mode = "100644"
        type = "blob"
        sha = $blobResult.sha
    }
    $treeItems += $treeItem
}

# 5. 创建新的树
Write-Host "\n4. 创建新的树..." -ForegroundColor Yellow
$newTreeUrl = "https://api.github.com/repos/$OWNER/$REPO/git/trees"
$newTreeData = @{
    base_tree = $currentTreeSha
    tree = $treeItems
}
$newTreeResult = Invoke-RestMethod -Uri $newTreeUrl -Headers $headers -Method Post -Body ($newTreeData | ConvertTo-Json)
$newTreeSha = $newTreeResult.sha
Write-Host "✅ 新树 SHA: $newTreeSha" -ForegroundColor Green

# 6. 创建新的提交
Write-Host "\n5. 创建新的提交..." -ForegroundColor Yellow
$newCommitUrl = "https://api.github.com/repos/$OWNER/$REPO/git/commits"
$newCommitData = @{
    message = "Add server/dist directory (built backend)"
    parents = @($latestCommitSha)
    tree = $newTreeSha
}
$newCommitResult = Invoke-RestMethod -Uri $newCommitUrl -Headers $headers -Method Post -Body ($newCommitData | ConvertTo-Json)
$newCommitSha = $newCommitResult.sha
Write-Host "✅ 新提交 SHA: $newCommitSha" -ForegroundColor Green

# 7. 更新分支引用
Write-Host "\n6. 更新分支 $BRANCH 引用..." -ForegroundColor Yellow
$updateRefUrl = "https://api.github.com/repos/$OWNER/$REPO/git/refs/heads/$BRANCH"
$updateRefData = @{
    sha = $newCommitSha
    force = $true
}
$updateRefResult = Invoke-RestMethod -Uri $updateRefUrl -Headers $headers -Method Patch -Body ($updateRefData | ConvertTo-Json)
Write-Host "✅ 分支引用更新成功！" -ForegroundColor Green

Write-Host "\n🎉 所有文件上传完成！" -ForegroundColor Green
Write-Host "GitHub 仓库: https://github.com/$OWNER/$REPO/tree/$BRANCH" -ForegroundColor Cyan
Write-Host "提交信息: Add server/dist directory (built backend)" -ForegroundColor Cyan
Write-Host "提交 SHA: $newCommitSha" -ForegroundColor Cyan