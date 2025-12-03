# Claude AI 使用入门

## 环境检查

系统已成功安装：
- ✅ Node.js v24.11.1
- ✅ npm 11.6.2
- ✅ ZCF v3.4.1
- ✅ Claude AI 框架

## 初始化Claude AI

由于交互式命令在当前环境受限，您需要在**本地命令行终端**中执行以下步骤：

### 步骤1：打开命令提示符
1. 按下 `Win + R` 键
2. 输入 `cmd` 并按回车
3. 使用 `cd` 命令导航到项目目录：
   ```bash
   cd C:\Users\霍冠华\Documents\trae_projects\claude code
   ```

### 步骤2：初始化配置
在命令提示符中运行：
```bash
start_zcf.bat init
```

**配置过程说明：**

| 配置项 | 推荐选择 | 说明 |
|--------|----------|------|
| API类型 | api_key | 推荐使用API密钥方式 |
| API密钥 | 您的密钥 | 输入您的Claude API密钥 |
| AI模型 | claude-sonnet-4-5 | 最新且功能强大的模型 |
| 语言偏好 | zh-CN | 中文界面 |
| 代码类型 | claude-code | 专为代码生成优化 |

## 启动Claude AI

### 方法1：交互式菜单
```bash
start_zcf.bat
```

这会显示一个功能菜单，您可以通过键盘选择：
1. 代码生成
2. 代码分析
3. 文档生成
4. 单元测试生成
5. 代码优化

### 方法2：直接执行功能

```bash
# 生成代码
start_zcf.bat code

# 分析代码
start_zcf.bat analyze

# 生成文档
start_zcf.bat doc

# 生成单元测试
start_zcf.bat test

# 优化代码
start_zcf.bat optimize
```

## 使用示例

### 示例1：生成代码

**需求**：生成一个Python函数，实现快速排序算法

**操作步骤**：
1. 运行：`start_zcf.bat code`
2. 输入需求：`生成一个Python快速排序算法函数，包含注释和示例用法`
3. AI会返回完整的代码实现

### 示例2：分析代码

**需求**：分析一段JavaScript代码的问题

**操作步骤**：
1. 运行：`start_zcf.bat analyze`
2. 粘贴您的代码
3. AI会分析代码并提供改进建议

### 示例3：生成文档

**需求**：为一个Python类生成文档

**操作步骤**：
1. 运行：`start_zcf.bat doc`
2. 粘贴您的Python类代码
3. AI会生成完整的文档字符串

## 配置管理

### 更新配置
```bash
start_zcf.bat update
```

### 检查用量
```bash
start_zcf.bat ccu
```

### 更新系统
```bash
start_zcf.bat check-updates
```

## 常见问题

### 1. API密钥在哪里获取？
- 访问 Anthropic 官网 (https://www.anthropic.com/)
- 注册或登录账号
- 进入控制台创建API密钥

### 2. 支持哪些AI模型？
- claude-3-sonnet-20240229
- claude-3-opus-20240229
- claude-3-haiku-20240307
- claude-sonnet-4-5

### 3. 如何更换API密钥？
```bash
start_zcf.bat init
```
重新运行初始化命令并输入新的API密钥

## 技术支持

如果遇到问题，请检查：
1. 网络连接是否正常
2. API密钥是否有效
3. Node.js版本是否兼容

您也可以查看详细日志文件获取更多信息。

---

祝您使用愉快！