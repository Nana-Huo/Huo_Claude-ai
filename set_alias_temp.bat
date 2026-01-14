@echo off
reg add "HKCU\Software\Microsoft\Command Processor" /v Autorun /t REG_SZ /d "doskey 小霍=claude $* & doskey hgh=claude $*" /f
if %errorlevel% equ 0 (
    echo Permanent aliases set!
    echo Restart CMD to take effect.
) else (
    echo Failed! Run as administrator.
)