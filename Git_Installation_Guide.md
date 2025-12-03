# Git 和 Git Bash 安装配置指南

## 1. 下载 Git for Windows

请访问以下链接下载适合您系统的 Git 安装程序：

**官方下载链接：** [Git for Windows](https://git-scm.com/download/win)

> 系统将自动检测您的 Windows 版本（32位或64位）并提供合适的下载选项。

## 2. 安装 Git for Windows

下载完成后，双击运行安装程序，按照以下步骤进行安装：

### 步骤 1：选择安装路径
- 默认安装路径：`C:\Program Files\Git`
- 建议使用默认路径，除非您有特殊需求

### 步骤 2：选择组件
- 确保勾选 "Git Bash Here" 和 "Git GUI Here"
- 建议勾选 "Add a Git Bash Profile to Windows Terminal"

### 步骤 3：选择默认编辑器
- 选择您熟悉的文本编辑器（如 VS Code、Notepad++ 或 Vim）
- 如果不确定，选择 "Use Notepad++ as Git's default editor"

### 步骤 4：调整 PATH 环境变量
- 选择 "Git from the command line and also from 3rd-party software"
- 这将确保 Git 在所有终端中都可用

### 步骤 5：选择 HTTPS 传输后端
- 选择 "Use the OpenSSL library"
- 这是默认且推荐的选项

### 步骤 6：配置行尾转换
- 选择 "Checkout Windows-style, commit Unix-style line endings"
- 这是跨平台开发的最佳选择

### 步骤 7：配置终端模拟器
- 选择 "Use Windows' default console window"
- 或选择 "Use MinTTY"（提供更完整的 Unix 终端体验）

### 步骤 8：配置额外选项
- 建议勾选 "Enable file system caching" 和 "Enable Git Credential Manager"

### 步骤 9：安装完成
- 点击 "Finish" 完成安装

## 3. 验证 Git 安装

安装完成后，打开命令提示符或 PowerShell，运行以下命令验证安装：

```bash
git --version
```

如果安装成功，您将看到类似以下输出：
```
git version 2.45.2.windows.1
```

## 4. 配置 Git 基本信息

首次使用 Git 前，需要配置您的用户名和邮箱：

```bash
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱地址"
```

> 注意：用户名和邮箱将出现在您的 Git 提交记录中

## 5. 启动 Git Bash

您可以通过以下方式启动 Git Bash：

1. **开始菜单**：在开始菜单中搜索 "Git Bash"
2. **右键菜单**：在任意文件夹中右键点击，选择 "Git Bash Here"
3. **命令提示符**：在命令提示符中输入 `git bash`

## 6. Git Bash 基本配置

### 6.1 配置别名（可选）

为常用命令设置别名可以提高工作效率：

```bash
git config --global alias.st status
git config --global alias.ci commit
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.logg "log --oneline --graph --decorate --all"
```

### 6.2 配置颜色（可选）

启用 Git 命令输出的颜色显示：

```bash
git config --global color.ui true
```

## 7. 常见问题解决

### 问题 1：Git 命令不被识别

如果在命令提示符中运行 `git` 命令显示 "命令未找到"，请检查：
- Git 是否正确安装
- PATH 环境变量是否包含 Git 安装路径

### 问题 2：Git Bash 中文显示乱码

在 Git Bash 中运行以下命令解决中文乱码问题：

```bash
git config --global core.quotepath false
echo "export LANG='zh_CN.UTF-8'" >> ~/.bashrc
echo "export LC_ALL='zh_CN.UTF-8'" >> ~/.bashrc
```

然后重启 Git Bash。

## 8. 学习资源

- [Git 官方文档](https://git-scm.com/doc)
- [Pro Git 中文版](https://git-scm.com/book/zh/v2)
- [GitHub Git 教程](https://guides.github.com/introduction/git-handbook/)

## 9. 联系支持

如果您在安装或使用过程中遇到问题，可以：
- 查看 [Git for Windows 常见问题](https://gitforwindows.org/#faq)
- 在 [Stack Overflow](https://stackoverflow.com/questions/tagged/git) 上搜索相关问题

---

**安装完成后，您就可以开始使用 Git 和 Git Bash 进行版本控制了！**