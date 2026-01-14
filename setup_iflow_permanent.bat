@echo off
chcp 65001 >nul
echo ========================================
echo   设置永久 iflow (小霍) 命令
echo ========================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✓ 检测到管理员权限
    echo.
) else (
    echo ⚠️  未检测到管理员权限
    echo 将设置用户级别的别名（无需管理员权限）
    echo.
)

:: 运行 PowerShell 脚本
powershell -ExecutionPolicy Bypass -File "%~dp0setup_permanent_iflow.ps1"

echo.
echo ========================================
echo 按任意键退出...
pause >nul