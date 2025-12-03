# AI系统使用指南

## 一、系统概述

您已经成功搭建了**ZCF (Zero-Config Code Flow) v3.4.1**系统，这是一个与Claude Code集成的AI开发工具，可以帮助您使用AI进行代码生成、分析和管理。

## 二、基本使用步骤

### 1. 启动ZCF

在当前目录下运行：
```bash
.\start_zcf.bat
```

或者直接运行zcf命令（需要Node.js环境）：
```bash
npx zcf
```

### 2. 初始化配置

首次使用需要初始化Claude Code配置：
```bash
.\start_zcf.bat init
# 或
npx zcf init
```

初始化过程中，您需要：
- 选择API类型（auth_token, api_key, ccr_proxy）
- 输入API密钥或配置代理
- 选择API模型（如claude-sonnet-4-5）
- 配置输出语言等参数

### 3. 基本命令

| 命令 | 描述 |
|------|------|
| `.\start_zcf.bat` | 显示交互式菜单（默认） |
| `.\start_zcf.bat init` | 初始化Claude Code配置 |
| `.\start_zcf.bat update` | 仅更新工作流相关配置 |
| `.\start_zcf.bat ccr` | 配置模型路由代理 |
| `.\start_zcf.bat ccu` | 运行Claude Code用量分析 |
| `.\start_zcf.bat check-updates` | 检查并更新到最新版本 |

## 三、使用AI进行代码生成

### 1. 通过ZCF菜单使用

运行`.\start_zcf.bat`后，会显示交互式菜单，您可以：
- 选择代码生成选项
- 输入您的需求
- AI会生成相应的代码

### 2. 直接使用Claude Code

Claude Code是一个VS Code扩展，您可以在VS Code中：

1. 打开VS Code
2. 安装Claude Code扩展
3. 在编辑器中右键点击，选择"Claude Code"相关选项
4. 输入您的代码需求

### 3. 常见代码生成场景

**生成函数**：
```
请生成一个计算斐波那契数列的JavaScript函数，使用递归和迭代两种方式
```

**代码优化**：
```
请优化以下代码，使其更高效：
function calculateSum(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}
```

**错误修复**：
```
这段代码有什么错误？如何修复？
function factorial(n) {
    if (n = 0) {
        return 1;
    }
    return n * factorial(n - 1);
}
```

## 四、高级功能

### 1. 配置模型路由

使用CCR（Claude Code Router）配置模型代理：
```bash
.\start_zcf.bat ccr
```

### 2. 检查用量

查看Claude Code的使用情况：
```bash
.\start_zcf.bat ccu          # 每日用量
.\start_zcf.bat ccu monthly  # 月度用量
.\start_zcf.bat ccu --json   # JSON格式输出
```

### 3. 更新系统

保持系统最新：
```bash
.\start_zcf.bat check-updates
# 或
npx zcf check-updates
```

## 五、故障排除

### 1. 常见错误

- **命令未找到**：确保Node.js和npm已正确安装
- **API错误**：检查API密钥或代理配置是否正确
- **乱码问题**：确保系统语言设置正确

### 2. 重新安装

如果遇到严重问题，可以重新初始化：
```bash
.\start_zcf.bat uninstall
.\start_zcf.bat init
```

## 六、最佳实践

1. **明确需求**：向AI提供清晰、具体的需求描述
2. **代码审查**：生成的代码需要进行人工审查和测试
3. **持续学习**：了解AI的能力边界，合理使用
4. **版本控制**：对生成的代码使用Git进行版本管理

## 七、联系方式

如果遇到无法解决的问题，可以：
- 查看官方文档
- 检查更新版本
- 提交Issue到相关仓库

---

祝您使用愉快！