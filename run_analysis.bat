@echo off
echo 开始安装依赖包...
pip install -r requirements.txt
echo.
echo 开始运行数据分析...
python data_analysis.py
pause