# MongoDB安装指南

## 为什么需要MongoDB
MongoDB是舞蹈约课平台的数据库，用于存储用户信息、课程数据、预约记录等重要信息。

## 下载MongoDB
1. 访问MongoDB官网下载页面：https://www.mongodb.com/try/download/community
2. 选择版本：
   - Version: Community Server (最新稳定版)
   - OS: Windows
   - Package: MSI

## 安装MongoDB

### 步骤1：运行安装程序
双击下载的MSI文件开始安装。

### 步骤2：选择安装类型
- 选择「Custom」(自定义安装)
- 点击「Next」

### 步骤3：选择安装路径
- 默认路径：`C:\Program Files\MongoDB\Server\5.0\`
- 建议保持默认路径
- 点击「Next」

### 步骤4：配置服务
- 勾选「Install MongoDB as a Service」
- 勾选「Run service as Network Service user」
- 勾选「Start MongoDB Service when Windows starts」
- 数据目录：`C:\data\db`
- 日志目录：`C:\data\log`
- 点击「Next」

### 步骤5：完成安装
- 点击「Install」
- 等待安装完成
- 点击「Finish」

## 验证MongoDB安装

### 方法1：检查服务状态
1. 按下 `Win + R`，输入 `services.msc`
2. 在服务列表中找到「MongoDB Server」
3. 确认状态为「Running」

### 方法2：命令行验证
1. 打开命令提示符
2. 输入 `mongo --version`
3. 如果显示MongoDB版本信息，说明安装成功

## 配置环境变量（可选）
1. 将MongoDB的bin目录添加到环境变量：`C:\Program Files\MongoDB\Server\5.0\bin`
2. 这样可以在任意目录使用 `mongo` 和 `mongod` 命令

## 启动MongoDB服务

### 方法1：通过服务管理
```bash
# 启动服务
net start MongoDB

# 停止服务
net stop MongoDB
```

### 方法2：手动启动
```bash
# 创建数据目录
mkdir -p C:\data\db

# 启动MongoDB
mongod --dbpath "C:\data\db"
```

## 测试MongoDB连接
1. 打开命令提示符
2. 输入 `mongo`
3. 如果连接成功，会显示MongoDB Shell界面
4. 输入 `show dbs` 查看数据库列表
5. 输入 `exit` 退出

## 常见问题解决

### 问题1：数据目录不存在
解决方案：
```bash
mkdir -p C:\data\db
```

### 问题2：端口被占用
解决方案：
```bash
# 查看端口占用
netstat -ano | findstr :27017

# 使用不同端口启动
mongod --dbpath "C:\data\db" --port 27018
```

### 问题3：服务启动失败
解决方案：
- 检查数据目录权限
- 检查日志文件 `C:\data\log\mongod.log`
- 确保没有其他MongoDB实例在运行

## 继续项目启动
安装完成MongoDB后，回到舞蹈约课平台项目，执行以下命令：

```bash
# 启动开发服务器
cd dance-booking-app
npm run dev
```

## 更多信息
- MongoDB官方文档：https://docs.mongodb.com/manual/
- MongoDB中文文档：https://www.mongodb.org.cn/tutorial/