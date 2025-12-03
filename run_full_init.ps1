# Claude Code 完整初始化脚本
# 该脚本使用Start-Process处理ZCF的交互式输入

# 设置ZCF路径
$zcfPath = "$env:ProgramFiles\nodejs\node.exe"
$zcfArgs = "$env:APPDATA\npm\node_modules\zcf\bin\zcf.mjs"

# 创建一个临时文件用于输入
$inputFile = "$PSScriptRoot\zcf_init_temp.txt"

# 写入完整初始化所需的所有输入
# 1. 选择完整初始化(选项1)
# 2. 保持默认模板语言配置(选项no)
"1`nno" | Out-File -FilePath $inputFile -Encoding UTF8

# 执行ZCF并重定向输入
$process = Start-Process -FilePath $zcfPath -ArgumentList $zcfArgs -RedirectStandardInput $inputFile -NoNewWindow -Wait -PassThru

# 清理临时文件
Remove-Item -Path $inputFile -Force

# 显示结果
Write-Host "ZCF完整初始化完成，退出代码: $($process.ExitCode)"