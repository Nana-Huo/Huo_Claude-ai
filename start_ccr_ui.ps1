# 设置Node.js路径
$nodePath = "C:\Program Files\nodejs\node.exe"

# 设置CCR CLI的完整路径
$ccrCliPath = "C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@musistudio\claude-code-router\dist\cli.js"

# 检查node.exe是否存在
if (-not (Test-Path $nodePath)) {
    Write-Error "Node.js not found at $nodePath"
    pause
    exit 1
}

# 检查ccr cli.js是否存在
if (-not (Test-Path $ccrCliPath)) {
    Write-Error "CCR CLI not found at $ccrCliPath"
    pause
    exit 1
}

Write-Host "Starting CCR UI..."

# 使用完整路径启动CCR UI
& $nodePath $ccrCliPath ui