# Claude AI 项目打包与部署指南

## 项目概述
这是一个基于 Claude AI 的企业级智能助手项目，提供命令行和网站两种使用方式，支持企业知识库导入和智能问答功能。

## 项目结构

```
.
├── .gitignore              # Git忽略文件配置
├── .spec-workflow/         # 项目规范工作流
├── AI_Usage_Guide.md       # AI使用指南
├── ALIAS_SETUP_GUIDE.md    # 别名设置指南
├── CMD_ALIAS_MANUAL.txt    # CMD别名手动设置说明
├── Claude使用入门.md       # Claude使用入门指南
├── GitHub推送与Vercel部署指南.md  # GitHub和Vercel部署指南
├── Git_Installation_Guide.md      # Git安装指南
├── LICENSE                 # MIT许可证
├── MongoDB安装指南.md      # MongoDB安装指南
├── README.md               # 项目说明文档
├── VERCEL_FIX_GUIDE.md     # Vercel部署修复指南
├── alias.cmd               # 简化的别名设置脚本
├── setup_aliases.ps1       # PowerShell别名设置脚本
├── setup_cmd_aliases.bat   # CMD别名设置脚本
└── dance-booking-app/      # 示例应用（可忽略）
```

## 打包步骤

### 1. 清理不必要的文件

首先，确保项目中只包含必要的文件。如果您想要专注于Claude AI助手功能，可以考虑移除dance-booking-app目录：

```bash
# PowerShell
Remove-Item -Recurse -Force dance-booking-app
```

### 2. 创建可执行的安装包

#### Windows 命令行工具打包

1. **创建安装脚本**：
   ```bash
   # 创建 install_claude.cmd
   @echo off
   echo 正在安装 Claude AI 助手...
   echo 请确保已安装 Node.js 和 npm
   
   # 设置别名
   reg add "HKEY_CURRENT_USER\Software\Microsoft\Command Processor" /v "Autorun" /t REG_SZ /d "doskey 小宝=claude $* & doskey hgh=claude $*" /f
   
   echo 安装完成！
   echo 请重新打开 CMD 窗口，然后输入 '小宝' 或 'hgh' 开始使用 Claude AI 助手
   pause
   ```

2. **创建压缩包**：
   - 选中所有Claude AI相关文件
   - 右键选择「发送到」→「压缩(zipped)文件夹」
   - 命名为 `Claude-AI-Assistant.zip`

#### Web 界面打包

如果您想要提供Web界面版本，需要：

1. 创建一个简单的Web界面
2. 配置Claude API集成
3. 使用打包工具（如Webpack、Vite）构建静态文件
4. 提供部署说明

## GitHub 上传步骤

### 1. 创建 GitHub 仓库

1. 登录 GitHub
2. 点击「New repository」
3. 输入仓库名称（如 `claude-ai-assistant`）
4. 选择「Public」或「Private」
5. 点击「Create repository」

### 2. 推送本地代码到 GitHub

```bash
# 添加远程仓库
git remote add origin <your-github-repo-url>

# 推送代码
git push -u origin master
```

### 3. 配置 GitHub 仓库

1. **添加项目描述**：在仓库首页点击「About」编辑项目描述
2. **设置开源许可证**：已添加MIT许可证
3. **创建发布版本**：
   - 点击「Releases」→「Draft a new release」
   - 填写版本号（如 `v1.0.0`）
   - 上传打包好的压缩包
   - 填写发布说明
   - 点击「Publish release」

## 使用方式

### 命令行使用（推荐）

1. **安装 Claude CLI**：
   ```bash
   npm install -g claude-code
   ```

2. **设置 API 密钥**：
   ```bash
   claude setup-token
   ```

3. **使用别名**：
   ```bash
   # 使用临时别名
   doskey 小宝=claude $*
   doskey hgh=claude $*
   
   # 使用永久别名
   # 运行 setup_cmd_aliases.bat 或 setup_aliases.ps1
   ```

4. **测试别名**：
   ```bash
   小宝 --version
   hgh "帮我写一个Python函数"
   ```

### Web 界面使用

1. **部署到 Vercel**：
   ```bash
   # 使用 Vercel CLI
   npm install -g vercel
   vercel login
   vercel deploy
   ```

2. **本地运行**：
   ```bash
   # 如果有Web界面代码
   npm install
   npm run dev
   ```

## 企业知识库功能

### 导入企业知识

1. 创建 `company-knowledge` 目录
2. 将企业文档放入该目录
3. 运行导入命令：
   ```bash
   claude import-docs --path "company-knowledge/"
   ```

### 更新知识库

```bash
claude update-knowledge
```

## 后续开发计划

1. **增强学习能力**：实现更智能的企业知识学习机制
2. **多平台支持**：扩展到Linux和macOS
3. **Web界面优化**：提供更友好的企业知识库管理界面
4. **权限管理**：添加企业级权限控制功能
5. **集成扩展**：支持与企业内部系统集成

## 故障排除

### 别名无法识别
- 确保已正确运行别名设置脚本
- 重启命令行窗口
- 检查注册表中的Autorun项是否正确设置

### Claude API 连接失败
- 检查网络连接
- 验证API密钥是否正确
- 查看错误日志获取详细信息

### Web 界面部署问题
- 确保所有依赖已正确安装
- 检查环境变量配置
- 查看Vercel部署日志

## 联系方式

如有任何问题或建议，请通过GitHub Issues反馈：
https://github.com/<your-username>/<your-repo>/issues

---

**Claude AI 助手** - 让企业知识触手可及！