---
name: office-utils
description: "办公文档处理工具，支持 Excel/Word/PDF/CSV 文件。适用场景：(1) 读取分析 Excel (.xlsx/.xls) 文件 (2) Excel 结构分析和字段类型推断 (3) 按类别分组统计汇总 (4) 读取 Word (.docx) 文档 (5) 提取 PDF 文本和表格 (6) 读取 CSV 文件（自动检测编码）(7) 写入 Excel/Word/CSV 文件 "
license: MIT. 完整条款见 LICENSE.txt
---

# Office Utils - 办公文档处理

## ⚡ 自动执行规则（必读）

**激活此 Skill 后，AI 必须立即自动执行，不要等待用户指示：**

### 第一步：扫描文件
扫描当前工作目录的所有办公文档（Excel、Word、PDF、CSV）

### 第二步：深度分析每个文件

**Excel 文件分析：**
- Sheet 角色（汇总表/明细表/模板/线索表）
- 表头结构（多行表头识别）
- 字段类型（标识/金额/时间/名称/状态）
- 数据量和关键字段

**Word 文件分析：**
- 文档结构（标题、段落、表格）
- 主要内容摘要
- 关键信息提取

**PDF 文件分析：**
- 页数和结构
- 文本内容提取
- 表格数据提取

### 第三步：创建工作记录
在 `处理结果/` 目录创建工作日志，**必须包含所有文件的分析结果**

### 第四步：输出摘要
综合所有文档内容，推断任务意图

**禁止：**
- ❌ 问"需要我做什么"
- ❌ 问"需要继续吗"
- ❌ 只分析部分文件

**要求：**
- ✅ 激活即执行
- ✅ 分析所有类型的文档（Excel/Word/PDF/CSV）
- ✅ 所有分析结果必须写入工作记录
- ✅ 最后给简洁摘要

---

## 快速开始

```python
# 依赖检测
python3 -c "
for p in ['openpyxl','docx','pdfplumber','chardet']:
    try: __import__(p); print(f'✅ {p}')
    except: print(f'❌ {p}')
"

# 安装缺失依赖
pip install openpyxl python-docx pdfplumber chardet pandas
```

## 核心工作流

**使用 [office_utils.py](office_utils.py) 工具库（v2.0 完整版）：**

```python
import sys
sys.path.insert(0, 'skills/office-utils')
from office_utils import *
```

### 1. 扫描文件 - `scan_office_files()`

```python
files = scan_office_files('.')                    # 扫描当前目录
files = scan_office_files('.', recursive=True)   # 递归扫描子目录
# 返回: [Path('file1.xlsx'), Path('file2.docx'), ...]
```

### 2. 获取文件信息（快速，不读数据） - `get_file_info()`

```python
info = get_file_info('file.xlsx')
# 返回: {
#   'success': True,
#   'data': {
#     'name': 'file.xlsx', 'type': 'excel', 'size_human': '1.2MB',
#     'structure': {'sheets': [{'name': 'Sheet1', 'rows': 1000, 'actual_rows': 856}]}
#   }
# }
```

### 3. 读取 Excel - `read_excel()`

```python
result = read_excel('file.xlsx')                           # 读取所有 sheet
result = read_excel('file.xlsx', sheet_name='Sheet1')      # 指定 sheet
result = read_excel('file.xlsx', preview_rows=10)          # 只读前10行
result = read_excel('file.xlsx', header_rows=2)            # 多行表头
result = read_excel('file.xlsx', handle_merged=True)       # 处理合并单元格
# 返回: {'success': True, 'data': {'sheets': [{'name', 'headers', 'data', 'column_types', ...}]}}
```

### 4. 深度分析 Excel 结构 - `analyze_excel_structure()`

```python
analysis = analyze_excel_structure('file.xlsx')
# 返回: {
#   'overview': {'sheet_count': 3, 'total_data_rows': 1500},
#   'sheets': [{'name', 'role', 'headers', 'field_analysis', 'data_preview'}],
#   'task_hints': ['可能需要：数据筛选、统计分析'],
#   'key_fields': ['机构名称', '金额']
# }
# role 类型: 汇总表/明细表/模板/配置表/线索表
```

### 5. 分组统计 - `group_statistics()`

```python
result = group_statistics('file.xlsx')                           # 自动识别分组列和金额列
result = group_statistics('file.xlsx', group_by='机构名称')       # 指定分组列
result = group_statistics('file.xlsx', agg_cols=['金额', '数量']) # 指定聚合列
result = group_statistics('file.xlsx', top_n=10)                 # 返回前10组
# 返回: {'success': True, 'data': {'group_by', 'total_groups', 'total_amounts', 'top_groups', 'distribution'}}
```

### 6. 读取 Word - `read_word()`

```python
result = read_word('file.docx')
# 返回: {'success': True, 'data': {
#   'structure': {'paragraphs_count', 'tables_count', 'headings_count'},
#   'content': {'paragraphs', 'headings', 'tables'},
#   'outline': ['标题1', '  子标题1.1', ...]
# }}
```

### 7. 读取 PDF - `read_pdf()`

```python
result = read_pdf('file.pdf')                      # 读取全部页
result = read_pdf('file.pdf', pages=[0, 1])        # 只读第1、2页
result = read_pdf('file.pdf', extract_tables=True) # 提取表格
# 返回: {'success': True, 'data': {'structure': {'pages_count'}, 'pages': [{'page_num', 'text', 'tables'}]}}
```

### 8. 读取 CSV - `read_csv()`

```python
result = read_csv('file.csv')                      # 自动检测编码和分隔符
result = read_csv('file.csv', preview_rows=100)    # 只读前100行
result = read_csv('file.csv', encoding='gbk')      # 指定编码
# 返回: {'success': True, 'data': {'headers', 'data', 'encoding', 'delimiter'}}
```

### 9. 写入文件 - `write_excel()` / `write_csv()` / `write_word()`

```python
# 写入 Excel
write_excel(data, 'output.xlsx', headers=['列1', '列2'])
write_excel(data, 'output.xlsx', template_path='template.xlsx')  # 基于模板

# 写入 CSV (utf-8-sig 编码，Excel 友好)
write_csv(data, 'output.csv', headers=['列1', '列2'])

# 写入 Word
write_word({'title': '报告', 'paragraphs': ['段落1'], 'tables': [...]}, 'output.docx')
```

### 10. 目录分析 - `analyze_directory()`

```python
result = analyze_directory('.')                    # 分析当前目录
result = analyze_directory('.', parallel=True)     # 并行处理（多文件更快）
# 返回: {'success': True, 'data': {
#   'summary': {'total_files', 'by_type', 'total_data_rows', 'keywords', 'work_type_guess'},
#   'files': [{'name', 'type', 'structure', 'preview'}]
# }}
```


## 参考文档

- **[references/field-types.md](references/field-types.md)** - 字段类型识别规则和 Sheet 角色判断
- **[references/project-structure.md](references/project-structure.md)** - 项目目录结构和命名规范
- **[references/work-log-template.md](references/work-log-template.md)** - 工作日志模板和记录格式

## 脚本

- **[scripts/office_utils.py](scripts/office_utils.py)** - 完整工具库，可直接 import 使用
