@echo off
chcp 65001 >nul

echo Setting Claude Code CMD Aliases...

:: Create temporary aliases
doskey 小宝=claude $*
doskey hgh=claude $*

echo Aliases set!
echo.
echo Usage:
echo   小宝 --version
echo   hgh "Your question"
echo.
echo Note: These aliases work only in this CMD session.
echo.

set /p "choice=Set permanent aliases? (Y/N): "

if /i "%choice%"=="Y" (
    echo.
    echo Creating autorun registry entry...
    reg add "HKCU\Software\Microsoft\Command Processor" /v Autorun /t REG_SZ /d "doskey 小宝=claude $* & doskey hgh=claude $*" /f
    
    if %errorlevel% equ 0 (
        echo Permanent aliases set!
        echo Restart CMD to take effect.
    ) else (
        echo Failed! Run as administrator.
    )
)

echo.
echo Test after restarting CMD:
echo   小宝 --version
echo   hgh "Hello"

echo.
echo Press any key to continue...
pause >nul