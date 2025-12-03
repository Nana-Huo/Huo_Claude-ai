@echo off

REM 检查Git是否可用
echo 正在检查Git安装...
where git.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo Git未找到，正在添加Git到环境变量...
    set "PATH=%PATH%;C:\Program Files\Git\bin"
    where git.exe >nul 2>nul
    if %errorlevel% neq 0 (
        echo 无法找到Git，请检查安装路径。
        pause
        exit /b 1
    )
)

echo Git安装成功！版本：
git --version
echo.

REM 检查Git配置
echo 正在检查Git配置...
set "username="
for /f "tokens=2* delims= " %%i in ('git config --global user.name 2^>nul') do set "username=%%i"

set "email="
for /f "tokens=2* delims= " %%i in ('git config --global user.email 2^>nul') do set "email=%%i"

if not defined username (
    echo Git用户名未配置，正在配置默认值...
    git config --global user.name "霍冠华"
    set "username=霍冠华"
)

if not defined email (
    echo Git邮箱未配置，正在配置默认值...
    git config --global user.email "hgh@example.com"
    set "email=hgh@example.com"
)

echo Git配置信息：
echo 用户名：%username%
echo 邮箱：%email%
echo.

REM 启动Git Bash
echo 正在启动Git Bash...
start "Git Bash" "C:\Program Files\Git\git-bash.exe"

echo Git Bash已启动！
echo 您可以开始使用Git和Git Bash进行版本控制了。
echo.
echo 常用Git命令：
echo   git status     - 查看当前仓库状态
echo   git clone      - 克隆远程仓库
echo   git add        - 添加文件到暂存区
echo   git commit     - 提交更改
echo   git push       - 推送到远程仓库
echo   git pull       - 从远程仓库拉取
echo   git branch     - 查看分支
echo   git checkout   - 切换分支
echo.
pause