# Claude Code 别名配置指南

## 问题分析
从 Terminal#5-20 可以看到，您尝试了多次打开 PowerShell 配置文件并设置别名，但 `小宝` 命令仍然无法识别。这是因为我受到权限限制，无法直接修改您的 PowerShell 配置文件。

## 解决方案

### 方法一：临时别名（每次打开PowerShell都需要执行）

1. 打开 PowerShell
2. 执行以下命令：

```powershell
# 添加Claude Code别名
New-Alias -Name "小宝" -Value "claude" -Force
New-Alias -Name "hgh" -Value "claude" -Force

# 确保npm全局目录在PATH中
$env:PATH += ";C:\Program Files\nodejs"
$env:PATH += ";C:\Users\霍冠华\AppData\Roaming\npm"
```

### 方法二：使用我创建的脚本

1. 打开 PowerShell 并导航到项目目录：
```powershell
cd "C:\Users\霍冠华\Documents\trae_projects\claude code"
```

2. 执行我创建的脚本：
```powershell
.\setup_aliases.ps1
```

### 方法三：永久别名配置（推荐）

1. 打开 PowerShell 配置文件：
```powershell
notepad $PROFILE
```

2. 在打开的记事本中粘贴以下内容：
```powershell
# Claude Code 别名配置
New-Alias -Name "小宝" -Value "claude" -Force
New-Alias -Name "hgh" -Value "claude" -Force

# 确保npm全局目录在PATH中
$env:PATH += ";C:\Program Files\nodejs"
$env:PATH += ";C:\Users\霍冠华\AppData\Roaming\npm"
```

3. 保存并关闭记事本

4. 重新启动 PowerShell 或执行以下命令应用更改：
```powershell
. $PROFILE
```

## 验证配置

配置完成后，您可以执行以下命令验证别名是否正常工作：

```powershell
# 检查别名是否存在
Get-Alias 小宝
Get-Alias hgh

# 测试Claude Code命令
小宝 --version
```

## 使用示例

```powershell
# 启动Claude Code交互式界面
小宝

# 直接提问
小宝 "帮我写一个Python函数"

# 查看帮助
小宝 --help

# 使用hgh别名
hgh "如何使用Claude Code？"
```

## 常见问题

### 1. 别名仍然无法识别

请确保：
- Node.js 和 npm 已正确安装
- `C:\Program Files\nodejs` 和 `C:\Users\霍冠华\AppData\Roaming\npm` 在系统 PATH 中
- 已重新启动 PowerShell 或执行 `. $PROFILE`

### 2. Claude 命令无法识别

请重新安装 Claude Code：
```powershell
npm install -g @anthropic-ai/claude-code
```

## 注意事项

- 首次使用 Claude Code 时，您需要通过 `claude setup-token` 设置 API 令牌
- 所有命令都可以使用 `小宝` 或 `hgh` 别名代替 `claude`
- 如果遇到任何问题，请随时告诉我！
