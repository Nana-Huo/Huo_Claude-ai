# iFlow iPhone 应用

这是一个将iFlow AI功能封装为iPhone应用的项目，通过中间层API服务实现与iFlow的交互。

## 项目结构

```
iFlowApp/
├── Model.swift          # 数据模型定义
├── APIService.swift     # API服务类，与中间层通信
├── ContentView.swift    # 主界面，包含聊天功能
├── ViewModel.swift      # 应用状态管理
├── iFlowAppApp.swift    # 应用主入口
└── README.md            # 项目说明文档
```

## 前置条件

1. 已安装 Node.js (用于运行中间层API服务)
2. 已安装 Xcode (用于开发和运行iPhone应用)
3. 已安装 iFlow CLI (已在您的系统上配置)

## 设置步骤

### 1. 启动中间层API服务

```bash
# 进入中间层API服务目录
cd ../iflow-api-server

# 安装依赖
npm install

# 启动服务
npm start
```

服务将运行在 `http://localhost:4000`

### 2. 运行iPhone应用

1. 打开 Xcode
2. 点击 "Open a project or file"
3. 选择 `iFlowApp` 目录
4. 连接您的iPhone设备或选择模拟器
5. 点击运行按钮 (▶️) 启动应用

## 应用功能

### 1. 基本聊天
- 在输入框中输入命令，点击发送按钮
- 应用将通过中间层API服务与iFlow交互
- 响应将显示在聊天界面中

### 2. 智能体选择
- 点击右下角的智能体图标
- 在弹出的智能体列表中选择一个智能体
- 应用将切换到该智能体的上下文

### 3. 健康检查
- 应用启动时会自动检查中间层API服务的健康状态
- 如果服务不可用，会显示错误信息

## API端点

中间层API服务提供以下端点：

- `GET /health` - 健康检查
- `GET /agents` - 获取智能体列表
- `POST /execute` - 执行iFlow命令
- `POST /agent/execute/:name` - 使用特定智能体执行命令

## 故障排除

### 1. 无法连接到中间层API服务
- 确保服务已启动 (`npm start`)
- 检查服务是否运行在 `http://localhost:4000`
- 检查防火墙设置，确保端口4000已打开

### 2. 智能体列表无法加载
- 确保iFlow CLI已正确安装和配置
- 检查中间层API服务的日志输出

### 3. 命令执行失败
- 检查iFlow CLI的权限和配置
- 检查中间层API服务的日志输出
- 确保您的命令格式正确

## 注意事项

1. 本应用仅在本地网络环境中测试通过
2. 如需在生产环境中使用，需要配置HTTPS和身份验证
3. iPhone应用和中间层API服务需要运行在同一网络中

## 未来改进

- 添加身份验证和授权功能
- 支持HTTPS
- 添加更多智能体管理功能
- 支持历史记录和收藏功能
- 添加离线模式支持
