# 字段类型识别与 Sheet 角色判断

## 字段类型识别

| 类型 | 关键词 |
|------|--------|
| 标识 | 编号、编码、ID、序号、code |
| 金额 | 金额、费用、元、价格、合计、amount、price |
| 时间 | 时间、日期、年、月、date、time |
| 名称 | 姓名、名称、机构、单位、人员、name |
| 状态 | 状态、结果、是否、标志、status |
| 地区 | 地址、地区、省、市、县 |

```python
def infer_field_type(field_name):
    if not field_name:
        return '其他'
    f = str(field_name).lower()
    
    if any(k in f for k in ['编号', '编码', 'id', 'code', '序号']): return '标识'
    if any(k in f for k in ['金额', '费用', '元', '价格', 'amount', 'price', '合计']): return '金额'
    if any(k in f for k in ['时间', '日期', 'date', 'time', '年', '月']): return '时间'
    if any(k in f for k in ['姓名', '名称', '机构', '单位', 'name', '人员']): return '名称'
    if any(k in f for k in ['状态', '结果', '是否', 'status', '标志']): return '状态'
    if any(k in f for k in ['地址', '地区', '省', '市', '县']): return '地区'
    return '其他'
```

## Sheet 角色判断

| 角色 | 特征 |
|------|------|
| 线索表 | 名称含"疑点"、"问题"、"线索"、"异常" |
| 汇总表 | 数据少（≤10行）、有合计行 |
| 明细表 | 数据多（>50行） |
| 模板表 | 几乎无数据（≤2行），仅有表头 |
| 数据表 | 其他情况（10-50行） |

```python
def infer_sheet_role(actual_rows, sheet_name):
    name = sheet_name.lower()
    
    # 关键词优先
    if any(k in name for k in ['疑点', '问题', '线索', '异常']): return '线索表'
    if any(k in name for k in ['汇总', '统计', 'summary']): return '汇总表'
    if any(k in name for k in ['模板', 'template']): return '模板表'
    
    # 行数判断
    if actual_rows <= 2: return '模板表'
    if actual_rows <= 10: return '汇总表'
    if actual_rows > 50: return '明细表'
    return '数据表'
```

## 处理建议

| 角色 | 处理方式 |
|------|----------|
| 汇总表 | 核对合计数据，检查公式 |
| 明细表 | 筛选、分组统计 |
| 模板表 | 用于生成输出文件 |
| 线索表 | 逐条核查，标注处理结果 |
