@echo off

REM 添加Git到当前终端环境变量
set "PATH=%PATH%;C:\Program Files\Git\bin"

REM 配置Git基本信息（使用默认值）
git config --global user.name "霍冠华"
git config --global user.email "hgh@example.com"

REM 配置Git别名
git config --global alias.st status
git config --global alias.ci commit
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.logg "log --oneline --graph --decorate --all"

REM 配置颜色
git config --global color.ui true

REM 配置自动换行
git config --global core.autocrlf true

REM 配置凭证管理
git config --global credential.helper manager

echo Git配置完成！
echo 用户名：霍冠华
echo 邮箱：hgh@example.com
echo 别名：st, ci, co, br, logg
echo.

REM 显示当前Git配置
echo 当前Git配置：
git config --list

echo.
echo 正在启动Git Bash...
start "Git Bash" "C:\Program Files\Git\git-bash.exe"

pause