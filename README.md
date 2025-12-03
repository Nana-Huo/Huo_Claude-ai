# Claude AI 助手 - 企业知识库与智能问答系统

## 项目简介
这是一个基于 Claude AI 的企业级智能助手项目，旨在帮助企业快速构建自己的知识库问答系统，方便员工查询公司知识和进行日常办公辅助。

## 功能特性

### 🚀 核心功能
- **智能问答**：基于 Claude AI 的强大自然语言处理能力，支持多轮对话和复杂问题解答
- **企业知识库**：可导入和管理企业内部文档、制度、流程等知识
- **命令行别名**：提供便捷的 CMD/PowerShell 命令别名，快速访问 AI 助手
- **多平台支持**：
  - Windows 命令行（CMD/PowerShell）
  - Web 界面（可部署到 Vercel/Netlify 等平台）
  - 可扩展至其他平台

### 💡 特色功能
- **自定义别名**：支持设置个性化命令别名（如 "小宝"、"hgh"）快速调用
- **批量部署脚本**：提供完整的部署和配置脚本，一键部署到云端
- **企业定制**：支持根据企业需求进行功能扩展和定制开发
- **学习能力**：可持续学习企业知识，不断提升回答准确性

## 快速开始

### 环境要求
- Node.js 16+ 和 npm
- Git
- Claude API 密钥（需自行申请）

### 安装步骤

#### 1. 克隆项目
```bash
git clone <your-github-repo-url>
cd claude-ai-assistant
```

#### 2. 设置 API 密钥
首次使用时需要设置 Claude API 密钥：
```bash
claude setup-token
```

#### 3. 配置命令别名

##### Windows CMD
运行别名设置脚本：
```bash
setup_cmd_aliases.bat
```

##### Windows PowerShell
运行 PowerShell 别名设置：
```bash
.\setup_aliases.ps1
```

## 使用指南

### 命令行使用

#### 基本用法
```bash
# 查询版本
小宝 --version
# 或使用 hgh

# 直接提问
hgh "帮我写一个 Python 函数计算斐波那契数列"

# 启动交互式界面
小宝
```

### Web 界面使用

1. 部署到 Vercel（推荐）：
```bash
npm run deploy:vercel
```

2. 或本地启动：
```bash
npm run dev
```

3. 访问 http://localhost:3000

### 企业知识库管理

#### 导入知识文档
```bash
claude import-docs --path "company-docs/"
```

#### 更新知识库
```bash
claude update-knowledge
```

## 项目结构

```
.
├── config/                  # 配置文件目录
├── docs/                   # 文档和指南
├── scripts/                # 脚本文件
│   ├── setup_aliases.ps1   # PowerShell 别名设置
│   ├── setup_cmd_aliases.bat  # CMD 别名设置
│   └── deploy-to-vercel.ps1   # Vercel 部署脚本
├── web/                    # Web 界面代码
├── .gitignore              # Git 忽略文件
├── LICENSE                 # 许可证
└── README.md               # 项目说明
```

## 部署方案

### 1. 命令行工具部署
适合个人或小团队使用，直接在本地命令行运行。

### 2. Web 界面部署

#### Vercel 部署（推荐）
```bash
.\deploy-to-vercel.ps1
```

#### Netlify 部署
```bash
.\deploy-to-netlify.ps1
```

### 3. 企业内部部署
提供 Docker 镜像支持，可部署到企业内部服务器：
```bash
docker-compose up -d
```

## 企业定制

### 添加企业知识
1. 创建 `company-knowledge` 目录
2. 将企业文档（PDF、Markdown、Word等）放入该目录
3. 运行：
```bash
claude train --path "company-knowledge/"
```

### 定制化开发
项目采用模块化设计，支持：
- 添加新的命令别名
- 扩展问答功能
- 集成企业内部系统
- 开发自定义插件

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程
1. Fork 本仓库
2. 创建新分支：`git checkout -b feature/xxx`
3. 提交修改：`git commit -m "Add feature xxx"`
4. 推送分支：`git push origin feature/xxx`
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 支持与反馈

### 问题反馈
- 提交 GitHub Issue
- 联系项目维护者

### 常见问题

#### Q: 别名无法识别怎么办？
A: 请确保已正确运行别名设置脚本，并重启命令行窗口。

#### Q: 如何更新 Claude AI 版本？
A: 运行：`npm update -g claude-code`

#### Q: 企业知识学习效果不佳？
A: 尝试：
1. 增加更多结构化文档
2. 优化文档格式（使用 Markdown 更佳）
3. 定期更新知识库

## 版本历史

- v1.0.0 (2024-01-15): 初始版本发布
- v1.1.0 (2024-02-20): 添加 Web 界面支持
- v2.0.0 (2024-03-30): 企业知识库功能上线

## 致谢

感谢 Anthropic 提供的 Claude AI API，以及所有贡献者的支持！

---

**Claude AI 助手** - 让企业知识触手可及！