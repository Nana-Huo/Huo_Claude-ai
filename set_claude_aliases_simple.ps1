# Set Claude Code aliases
New-Alias -Name "小宝" -Value "claude" -Force
New-Alias -Name "hgh" -Value "claude" -Force

# Ensure npm global directory is in PATH
$env:PATH += ";C:\Program Files\nodejs"
$env:PATH += ";C:\Users\霍冠华\AppData\Roaming\npm"

Write-Host "Claude aliases set!"