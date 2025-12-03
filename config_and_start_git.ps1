# 检查Git是否安装
Write-Host "正在检查Git安装..."
$gitPath = "C:\Program Files\Git\bin\git.exe"
if (-not (Test-Path $gitPath)) {
    Write-Host "Git未找到，请检查安装路径。" -ForegroundColor Red
    Read-Host "按Enter键退出..."
    exit 1
}

# 添加Git到当前会话的环境变量
$env:PATH += ";C:\Program Files\Git\bin"

# 显示Git版本
Write-Host "Git安装成功！版本：" -ForegroundColor Green
& $gitPath --version
Write-Host

# 检查并配置Git用户名
Write-Host "正在检查Git配置..."
$username = & $gitPath config --global user.name 2>$null
if (-not $username) {
    Write-Host "Git用户名未配置，正在配置默认值..."
    & $gitPath config --global user.name "霍冠华"
    $username = "霍冠华"
}

# 检查并配置Git邮箱
$email = & $gitPath config --global user.email 2>$null
if (-not $email) {
    Write-Host "Git邮箱未配置，正在配置默认值..."
    & $gitPath config --global user.email "hgh@example.com"
    $email = "hgh@example.com"
}

# 显示配置信息
Write-Host "Git配置信息：" -ForegroundColor Green
Write-Host "用户名：$username"
Write-Host "邮箱：$email"
Write-Host

# 配置Git别名
Write-Host "正在配置Git别名..."
& $gitPath config --global alias.st status
& $gitPath config --global alias.ci commit
& $gitPath config --global alias.co checkout
& $gitPath config --global alias.br branch
& $gitPath config --global alias.logg "log --oneline --graph --decorate --all"
Write-Host "Git别名配置完成：st, ci, co, br, logg"
Write-Host

# 启动Git Bash
Write-Host "正在启动Git Bash..." -ForegroundColor Green
Start-Process "C:\Program Files\Git\git-bash.exe"

Write-Host "Git Bash已启动！" -ForegroundColor Green
Write-Host "您可以开始使用Git和Git Bash进行版本控制了。"
Write-Host
Write-Host "常用Git命令："
Write-Host "  git status     - 查看当前仓库状态"
Write-Host "  git clone      - 克隆远程仓库"
Write-Host "  git add        - 添加文件到暂存区"
Write-Host "  git commit     - 提交更改"
Write-Host "  git push       - 推送到远程仓库"
Write-Host "  git pull       - 从远程仓库拉取"
Write-Host "  git branch     - 查看分支"
Write-Host "  git checkout   - 切换分支"
Write-Host
Read-Host "按Enter键退出..."