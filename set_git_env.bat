@echo off

REM 添加Git到系统环境变量
setx PATH "%PATH%;C:\Program Files\Git\bin" /M

echo Git环境变量设置完成！
echo.
echo 请重启所有终端窗口以生效。
echo.

REM 配置Git基本信息
echo 正在配置Git基本信息...
echo 请输入您的Git用户名（将显示在提交记录中）：
set /p username=

if not defined username (
    echo 用户名不能为空，配置将使用默认值。
    set username=anonymous
)

echo 请输入您的Git邮箱（将显示在提交记录中）：
set /p email=

if not defined email (
    echo 邮箱不能为空，配置将使用默认值。
    set email=anonymous@example.com
)

git config --global user.name "%username%"
git config --global user.email "%email%"

echo.
echo Git基本信息配置完成！
echo 用户名：%username%
echo 邮箱：%email%
echo.

pause