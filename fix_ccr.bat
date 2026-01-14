@echo off
echo 正在修复CCR配置...
echo.
echo 请选择解决方案：
echo 1. 使用官方Anthropic API
echo 2. 修复iflow.cn配置
echo 3. 查看当前配置
echo.
set /p choice=请输入选项 (1-3): 

if "%choice%"=="1" (
    echo 正在配置官方Anthropic API...
    echo 请输入您的Anthropic API密钥:
    set /p api_key=API密钥: 
    echo.
    echo 设置环境变量...
    set ANTHROPIC_API_KEY=%api_key%
    echo API密钥已设置！
    echo.
    echo 现在请重新启动Claude Code进行测试
    pause
) else if "%choice%"=="2" (
    echo 启动CCR配置界面...
    .\start_zcf.bat ccr
) else if "%choice%"=="3" (
    echo 查看CCR配置...
    if exist "C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@musistudio\claude-code-router\config.json" (
        type "C:\Users\霍冠华\AppData\Roaming\npm\node_modules\@musistudio\claude-code-router\config.json"
    ) else (
        echo 未找到CCR配置文件
    )
    pause
) else (
    echo 无效选项
    pause
)