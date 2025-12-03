# GitHub代码推送与Vercel部署指南

## 一、当前项目状态

✅ 本地Git仓库已初始化
✅ 代码已提交到本地仓库（提交信息："Initial commit: 舞蹈约课平台项目"）
✅ .gitignore文件已创建
✅ 项目结构完整（client/和server/目录）

## 二、手动推送代码到GitHub步骤

### 1. 创建GitHub仓库

1. 登录GitHub（https://github.com）
2. 点击右上角"+"号，选择"New repository"
3. 仓库名称填写：`dance-booking-app`
4. 描述：`舞蹈约课平台 - 前后端一体化应用`
5. 选择"Public"或"Private"
6. 不勾选"Add a README file"等选项（我们已有本地README）
7. 点击"Create repository"

### 2. 配置SSH密钥（如果尚未配置）

#### 生成SSH密钥

```bash
# 打开Git Bash或命令提示符
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 按Enter键接受默认位置
# 输入密码（可选，直接按Enter跳过）
```

#### 查看并复制公钥

```bash
# Git Bash
cat ~/.ssh/id_rsa.pub

# Windows命令提示符
type %USERPROFILE%\.ssh\id_rsa.pub
```

#### 添加公钥到GitHub

1. 登录GitHub，点击右上角头像 → "Settings"
2. 点击左侧"SSH and GPG keys" → "New SSH key"
3. Title：填写"本地机器"或其他标识
4. Key：粘贴刚才复制的公钥
5. 点击"Add SSH key"

### 3. 推送本地代码到GitHub

```bash
# 进入项目目录
cd C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app

# 添加远程仓库（使用SSH协议）
git remote add origin git@github.com:Nana-Huo/dance-booking-app.git

# 推送代码并设置上游分支
git push -u origin master
```

## 三、部署到Vercel步骤

### 1. 准备工作

1. 确保代码已成功推送到GitHub
2. 访问Vercel官网（https://vercel.com）
3. 使用GitHub账号登录

### 2. 导入GitHub仓库

1. 点击"Add New" → "Project"
2. 在"Import Git Repository"中搜索"dance-booking-app"
3. 点击仓库名称进行导入

### 3. 配置项目

#### 前端配置（client目录）

- **Root Directory**: `client`
- **Framework Preset**: 选择"Vite"（因为项目使用Vite）
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

#### 环境变量

添加项目所需的环境变量（从`.env.example`复制并修改）：

- `VITE_API_BASE_URL`: `https://your-backend-api.com`（后端API地址）

### 4. 部署

点击"Deploy"按钮，Vercel将自动构建并部署项目。

部署完成后，你将获得一个Vercel提供的域名，例如：`dance-booking-app.vercel.app`

### 5. 后端部署（可选）

后端代码可以部署到Vercel的Serverless Functions或其他平台（如Render、Heroku等）。

#### Vercel Serverless Functions配置

1. 在项目根目录创建`vercel.json`文件：

```json
{
  "version": 2,
  "builds": [
    {
      "src": "client/package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "client/dist"
      }
    },
    {
      "src": "server/src/index.ts",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "server/src/index.ts"
    },
    {
      "src": "/(.*)",
      "dest": "client/dist/$1"
    }
  ]
}
```

2. 更新后端代码以适配Serverless环境

3. 重新部署项目

## 四、常见问题解决

### 1. Git推送失败

**问题**：`git@github.com: Permission denied (publickey)`

**解决方法**：
- 确保SSH密钥已正确添加到GitHub
- 检查本地密钥权限：`chmod 600 ~/.ssh/id_rsa`
- 测试SSH连接：`ssh -T git@github.com`

**问题**：`Failed to connect to github.com port 443`

**解决方法**：
- 检查网络连接
- 尝试使用HTTPS协议推送：
  ```bash
  git remote set-url origin https://github.com/Nana-Huo/dance-booking-app.git
  git push -u origin master
  ```

### 2. Vercel部署失败

**问题**：构建命令失败

**解决方法**：
- 确保`client/package.json`中存在`build`脚本
- 检查依赖是否正确安装：`npm install`

**问题**：环境变量缺失

**解决方法**：
- 在Vercel控制台中添加所有必要的环境变量
- 确保环境变量名称与代码中使用的一致

## 五、项目启动命令

### 本地开发

```bash
# 安装依赖
npm run install:all

# 启动开发服务器
npm run dev

# 前端访问地址：http://localhost:3000
# 后端API地址：http://localhost:5000
```

### 生产环境

```bash
# 构建前端
cd client
npm run build

# 启动后端
cd ../server
npm start
```

## 六、后续步骤

1. 完成GitHub代码推送
2. 完成Vercel部署
3. 测试在线应用
4. 配置域名（可选）
5. 监控应用性能

## 七、联系方式

如果在操作过程中遇到问题，可以随时咨询获取帮助。

---

**完成时间**：2025年1月13日
**项目路径**：C:\Users\霍冠华\Documents\trae_projects\claude code\dance-booking-app