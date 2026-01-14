# 可靠的iFlow启动脚本
# 这个脚本使用完整的绝对路径和PowerShell正确的语法，可以在任何终端运行

# 定义node.exe和iFlow CLI的完整路径
$nodePath = "C:\Program Files\nodejs\node.exe"
$iFlowPath = "C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@iflow-ai\iflow-cli\bundle\iflow.js"

# 检查node.exe是否存在
if (-not (Test-Path $nodePath)) {
    Write-Host "错误：找不到node.exe，请检查Node.js是否正确安装。"
    Write-Host "预期路径：$nodePath"
    pause
    exit 1
}

# 检查iFlow CLI是否存在
if (-not (Test-Path $iFlowPath)) {
    Write-Host "错误：找不到iFlow CLI，请检查iFlow是否正确安装。"
    Write-Host "预期路径：$iFlowPath"
    pause
    exit 1
}

# 使用正确的PowerShell语法启动iFlow
Write-Host "正在启动iFlow..."
& $nodePath $iFlowPath