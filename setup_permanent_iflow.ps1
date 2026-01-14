# 永久设置 iflow (小霍) 命令
# 以管理员身份运行此脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  设置永久 iflow (小霍) 命令" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否以管理员身份运行
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  警告: 此脚本需要管理员权限才能修改系统设置" -ForegroundColor Yellow
    Write-Host "请右键点击 PowerShell，选择 '以管理员身份运行'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "或者，我可以为您设置用户级别的别名（无需管理员权限）" -ForegroundColor Green
    $userChoice = Read-Host "是否继续设置用户级别别名？(Y/N)"
    
    if ($userChoice -ne "Y" -and $userChoice -ne "y") {
        Write-Host "已取消设置" -ForegroundColor Red
        exit
    }
}

Write-Host "开始设置..." -ForegroundColor Green
Write-Host ""

# 方法1: 添加到 PowerShell Profile
Write-Host "[1/3] 配置 PowerShell Profile..." -ForegroundColor Cyan

$profilePath = $PROFILE.CurrentUserCurrentHost
$profileDir = Split-Path $profilePath -Parent

# 创建 profile 目录（如果不存在）
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    Write-Host "  ✓ 创建 profile 目录: $profileDir" -ForegroundColor Green
}

# 检查是否已经添加了别名
$aliasContent = @'
# iflow (小霍) 命令别名
if (Get-Command claude -ErrorAction SilentlyContinue) {
    New-Alias -Name "小霍" -Value "claude" -Force -ErrorAction SilentlyContinue
    New-Alias -Name "iflow" -Value "claude" -Force -ErrorAction SilentlyContinue
    New-Alias -Name "hgh" -Value "claude" -Force -ErrorAction SilentlyContinue
}
'@

if (Test-Path $profilePath) {
    $currentProfile = Get-Content $profilePath -Raw
    if ($currentProfile -notmatch "小霍") {
        Add-Content -Path $profilePath -Value $aliasContent
        Write-Host "  ✓ 添加别名到 PowerShell Profile" -ForegroundColor Green
    } else {
        Write-Host "  ✓ 别名已存在于 PowerShell Profile" -ForegroundColor Green
    }
} else {
    Set-Content -Path $profilePath -Value $aliasContent
    Write-Host "  ✓ 创建新的 PowerShell Profile 并添加别名" -ForegroundColor Green
}

Write-Host "  Profile 位置: $profilePath" -ForegroundColor Gray
Write-Host ""

# 方法2: 创建 CMD 自动运行脚本
Write-Host "[2/3] 配置 CMD 自动运行..." -ForegroundColor Cyan

$cmdAutoRun = "doskey 小霍=claude $* & doskey iflow=claude $* & doskey hgh=claude $*"

if ($isAdmin) {
    # 系统级别
    reg add "HKCU\Software\Microsoft\Command Processor" /v Autorun /t REG_SZ /d $cmdAutoRun /f | Out-Null
    Write-Host "  ✓ 设置 CMD 自动运行（当前用户）" -ForegroundColor Green
} else {
    # 创建 doskey 宏文件
    $doskeyFile = "$env:USERPROFILE\Documents\cmd-aliases.txt"
    Set-Content -Path $doskeyFile -Value $cmdAutoRun
    Write-Host "  ✓ 创建 CMD 别名文件: $doskeyFile" -ForegroundColor Green
    Write-Host "    在 CMD 中运行: doskey /macrofile=$doskeyFile" -ForegroundColor Gray
}

Write-Host ""

# 方法3: 添加 npm 全局命令（如果 claude 是通过 npm 安装的）
Write-Host "[3/3] 检查 npm 全局安装..." -ForegroundColor Cyan

$claudePath = where.exe claude -ErrorAction SilentlyContinue
if ($claudePath) {
    Write-Host "  ✓ claude 命令已安装: $claudePath" -ForegroundColor Green
    
    # 尝试创建 iflow 包装脚本
    $npmGlobalPath = "$env:APPDATA\npm\iflow.cmd"
    $iflowScript = @"
@echo off
claude %*
"@
    
    try {
        Set-Content -Path $npmGlobalPath -Value $iflowScript -ErrorAction Stop
        Write-Host "  ✓ 创建全局命令: iflow.cmd" -ForegroundColor Green
        Write-Host "    位置: $npmGlobalPath" -ForegroundColor Gray
    } catch {
        Write-Host "  ⚠ 无法创建全局命令（权限不足）" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠ 未找到 claude 命令，请确保已安装 Claude Code CLI" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  设置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用方法：" -ForegroundColor White
Write-Host "  PowerShell: 小霍 '你好' 或 iflow '你好'" -ForegroundColor Gray
Write-Host "  CMD:       小霍 '你好' 或 iflow '你好'" -ForegroundColor Gray
Write-Host ""
Write-Host "注意：" -ForegroundColor Yellow
Write-Host "  • PowerShell 需要重新打开窗口或运行: . $PROFILE" -ForegroundColor Gray
Write-Host "  • CMD 需要重新打开窗口" -ForegroundColor Gray
Write-Host "  • 如果 claude 命令不存在，请先安装: npm install -g @anthropic-ai/claude-code" -ForegroundColor Gray
Write-Host ""