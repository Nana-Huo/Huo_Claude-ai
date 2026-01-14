# 设置Claude API密钥

# 用户提供的API密钥
$apiKey = "sk-4238b8abe1fcfa207b37a2d0443f0a8b"

# 设置环境变量（Claude会读取这个环境变量）
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", $apiKey, "User")

Write-Host "Claude API密钥已设置！"
Write-Host "密钥将在新的PowerShell会话中生效。"
Write-Host "您可以通过运行 'claude setup-token' 来验证和配置API密钥。"