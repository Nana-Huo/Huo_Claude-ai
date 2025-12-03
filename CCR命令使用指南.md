# CCR 命令使用指南

## 问题分析

您在尝试直接使用 `ccr code` 命令时遇到了以下错误：

```
ccr : 无法将“ccr”项识别为 cmdlet、函数、脚本文件或可运行程序的名称。
```

**错误原因**：
CCR（Claude Code Router）是作为 ZCF 系统的一部分提供的功能，而不是一个独立的命令。您需要通过 `start_zcf.bat` 脚本来调用 CCR 功能。

## 正确的 CCR 命令使用方式

### 方法1：通过 ZCF 交互式菜单

```bash
.\start_zcf.bat
```

在菜单中选择 "CCR 配置" 选项。

### 方法2：直接通过 ZCF 调用

```bash
.\start_zcf.bat ccr
```

### 方法3：完整命令格式

```bash
.\start_zcf.bat ccr <子命令> [参数]
```

## ZCF 支持的 CCR 相关命令

从 `start_zcf.bat --help` 的输出中，我们可以看到以下相关命令：

| 命令 | 功能 |
|------|------|
| `init` | 初始化 Claude Code 配置 |
| `update` | 更新 Claude Code 提示词 |
| `ccr` | 配置 Claude Code Router 用于模型代理 |
| `ccu [...args]` | 运行 Claude Code 使用分析工具 |
| `config-switch [target]` | 切换 Codex 提供商或 Claude Code 配置 |
| `uninstall` | 删除 ZCF 配置和工具 |
| `check-updates` | 检查并更新 Claude Code 和 CCR 到最新版本 |

## 常用 CCR 功能示例

### 配置模型代理
```bash
.\start_zcf.bat ccr
```

### 检查使用情况
```bash
.\start_zcf.bat ccu
```

### 更新系统
```bash
.\start_zcf.bat check-updates
```

### 切换配置
```bash
.\start_zcf.bat config-switch
```

## 解决命令未找到问题

### 1. 确保使用正确的命令格式
始终通过 `start_zcf.bat` 来调用功能，例如：
```bash
# 错误：ccr code
# 正确：.\start_zcf.bat ccr
```

### 2. 检查 ZCF 是否正确安装
```bash
.\start_zcf.bat --version
```

您应该看到类似输出：
```
ZCF - Zero-Config Code Flow v3.4.1
```

### 3. 更新系统到最新版本
```bash
.\start_zcf.bat check-updates
```

## 代码生成功能

如果您想生成代码，应该使用以下命令之一：

### 交互式代码生成
```bash
.\start_zcf.bat
```
然后选择 "代码生成" 选项。

### 直接代码生成
```bash
.\start_zcf.bat code
```

## 总结

CCR 是 ZCF 系统的一个功能模块，不是独立命令。请始终通过 `start_zcf.bat` 脚本来调用所有相关功能。

如果您有任何其他问题，请参考已创建的使用指南或随时询问。