# Claude Code 别名配置脚本
# 请以管理员身份运行此脚本或直接在PowerShell中执行以下命令

Write-Host "正在配置Claude Code别名..."

# 添加Claude Code别名
New-Alias -Name "小宝" -Value "claude" -Force
New-Alias -Name "hgh" -Value "claude" -Force

# 确保npm全局目录在PATH中
$env:PATH += ";C:\Program Files\nodejs"
$env:PATH += ";C:\Users\霍冠华\AppData\Roaming\npm"

Write-Host ""
Write-Host "别名配置完成！"
Write-Host "您现在可以使用以下命令："
Write-Host "  小宝 - 启动Claude Code"
Write-Host "  hgh - 启动Claude Code"
Write-Host ""
Write-Host "示例用法："
Write-Host "  小宝 "帮我写一个Python函数"
Write-Host "  hgh --version"
