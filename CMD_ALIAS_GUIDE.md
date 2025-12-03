# CMD 中 Claude Code 别名设置指南

## 概述

由于CMD（命令提示符）和PowerShell使用不同的别名机制，您需要使用专门的脚本在CMD中设置`小宝`和`hgh`别名。

## 区别说明

| 特性 | PowerShell | CMD |
|------|------------|-----|
| 别名命令 | `New-Alias` | `doskey` |
| 持久化方式 | 配置文件 `$PROFILE` | 注册表或自动运行脚本 |
| 命令参数传递 | 自动支持 | 需要 `$*` 显式指定 |

## 使用方法

### 方法一：临时别名（当前CMD会话有效）

1. 打开CMD命令提示符
2. 导航到项目目录：
   ```cmd
   cd "C:\Users\霍冠华\Documents\trae_projects\claude code"
   ```
3. 运行别名设置脚本：
   ```cmd
   setup_cmd_aliases.bat
   ```
4. 选择不设置永久别名（输入 `N`）

### 方法二：永久别名（所有CMD会话有效）

1. 以**管理员身份**打开CMD命令提示符
2. 导航到项目目录：
   ```cmd
   cd "C:\Users\霍冠华\Documents\trae_projects\claude code"
   ```
3. 运行别名设置脚本：
   ```cmd
   setup_cmd_aliases.bat
   ```
4. 选择设置永久别名（输入 `Y`）
5. 重新打开CMD窗口使设置生效

### 方法三：手动设置（高级用户）

1. 打开CMD命令提示符
2. 设置临时别名：
   ```cmd
   set "PATH=%PATH%;C:\Program Files\nodejs;C:\Users\霍冠华\AppData\Roaming\npm"
   doskey 小宝=claude $*
   doskey hgh=claude $*
   ```

## 使用示例

```cmd
:: 查看版本
小宝 --version

:: 直接提问
小宝 "帮我写一个Python函数"

:: 使用hgh别名
hgh "如何使用Claude Code？"

:: 查看帮助
hgh --help
```

## 检查别名

```cmd
:: 列出所有CMD别名
doskey /macros

:: 检查特定别名
doskey /macros | findstr "小宝 hgh"
```

## 移除别名

```cmd
:: 临时移除别名（当前会话有效）
doskey 小宝=
doskey hgh=

:: 永久移除别名
:: 删除自动运行脚本
del "%USERPROFILE%\cmd_aliases.bat"
:: 删除注册表项
reg delete "HKCU\Software\Microsoft\Command Processor" /v AutoRun /f
```

## 常见问题

### 1. 别名无法识别

确保：
- Node.js和npm已正确安装
- `C:\Program Files\nodejs` 和 `C:\Users\霍冠华\AppData\Roaming\npm` 在系统PATH中
- 对于永久别名，已重新打开CMD窗口

### 2. 运行脚本时出现权限错误

请以管理员身份运行CMD命令提示符。

### 3. Claude命令无法识别

请重新安装Claude Code：
```cmd
npm install -g @anthropic-ai/claude-code
```

## 注意事项

- 临时别名仅在当前CMD会话中有效
- 永久别名需要管理员权限设置
- CMD别名使用`doskey`命令，与PowerShell的`New-Alias`不同
- 别名配置会同时添加Node.js和npm路径到环境变量

如果您遇到任何问题，请随时联系技术支持！
