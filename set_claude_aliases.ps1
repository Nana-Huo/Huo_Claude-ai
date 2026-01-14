# Claude Code 别名配置

# 添加Claude Code别名
New-Alias -Name "小宝" -Value "claude" -Force
New-Alias -Name "hgh" -Value "claude" -Force

# 确保npm全局目录在PATH中
$env:PATH += ";C:\Program Files\nodejs"
$env:PATH += ";C:\Users\霍冠华\AppData\Roaming\npm"

Write-Host "Claude Code 别名已设置！"
Write-Host "您现在可以使用 '小宝' 或 'hgh' 来启动 Claude Code。"