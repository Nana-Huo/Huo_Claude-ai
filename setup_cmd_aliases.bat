@echo off
chcp 65001 >nul

:: 设置Claude Code CMD别名
echo 设置Claude Code CMD别名...

:: 创建临时别名（仅在当前CMD会话中有效）
doskey 小宝=claude $*
doskey hgh=claude $*

echo 别名设置完成！
echo.
echo 在CMD中使用以下命令：
echo.  小宝 --version
echo.  hgh "帮我写一个Python函数"
echo.
echo 注意：此别名仅在当前CMD会话中有效。
echo.

set /p "choice=是否要设置永久CMD别名？Y/N): "
if /i "%choice%"=="Y" goto PERMANENT
if /i "%choice%"=="N" goto END

echo 无效的选择，将不设置永久别名。
goto END

:PERMANENT
echo 创建自动运行注册表项...
reg add "HKCU\Software\Microsoft\Command Processor" /v Autorun /t REG_SZ /d "doskey 小宝=claude $* & doskey hgh=claude $*" /f
if %errorlevel% equ 0 (
    echo 永久别名设置完成！
    echo 重启CMD后别名将生效。
) else (
    echo 设置失败，请以管理员身份运行脚本。
)

echo.
echo 测试说明：
echo - 在当前脚本中，别名不会立即生效
echo - 请重新打开CMD窗口后测试以下命令：
echo   小宝 --version
echo   hgh "你好"

echo.
echo 请按任意键继续...
pause >nul

:END