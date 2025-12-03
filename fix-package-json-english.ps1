# GitHub API Configuration
$GITHUB_TOKEN = "github_pat_11AQX4B5A06BdY59b7Fb1bR58N3t3X5X9HxXaV4Bd3bR58N3t3X5X9HxXaV4Bd3"
$OWNER = "Nana-Huo"
$REPO = "dance-booking-app"
$BRANCH = "main"
$FILE_PATH = "package.json"

# Local file path
$LOCAL_FILE_PATH = "C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app\package.json"

Write-Host "Fixing and uploading $FILE_PATH to GitHub repository..." -ForegroundColor Green

# Set API headers
$headers = @{
    "Authorization" = "token $GITHUB_TOKEN"
    "Accept" = "application/vnd.github.v3+json"
}

# 1. Get current file SHA
Write-Host "\n1. Getting current SHA of $FILE_PATH in GitHub repository..." -ForegroundColor Yellow
$fileInfoUrl = "https://api.github.com/repos/$OWNER/$REPO/contents/$FILE_PATH?ref=$BRANCH"

try {
    $fileInfo = Invoke-RestMethod -Uri $fileInfoUrl -Headers $headers -Method Get
    $currentSha = $fileInfo.sha
    Write-Host "✅ Current file SHA: $currentSha" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to get file info: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 2. Read local file content
Write-Host "\n2. Reading local file content..." -ForegroundColor Yellow

try {
    $fileContent = Get-Content -Path $LOCAL_FILE_PATH -Raw -Encoding UTF8
    $base64Content = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($fileContent))
    Write-Host "✅ Successfully read local file and converted to Base64" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to read local file: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 3. Update file in GitHub repository
Write-Host "\n3. Updating $FILE_PATH in GitHub repository..." -ForegroundColor Yellow
$updateUrl = "https://api.github.com/repos/$OWNER/$REPO/contents/$FILE_PATH"
$updateData = @{
    message = "Fix package.json encoding issues and syntax errors"
    content = $base64Content
    sha = $currentSha
    branch = $BRANCH
} | ConvertTo-Json

try {
    $updateResult = Invoke-RestMethod -Uri $updateUrl -Headers $headers -Method Put -Body $updateData -ContentType "application/json"
    Write-Host "✅ File updated successfully!" -ForegroundColor Green
    Write-Host "Commit message: Fix package.json encoding issues and syntax errors" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Failed to update file: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "\n🎉 Fix completed! You can now redeploy on Vercel." -ForegroundColor Green