const express = require('express');
const { execSync } = require('child_process');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 4000;

// iFlow CLI路径配置
const NODE_PATH = '"C:\\Program Files\\nodejs\\node.exe"';
const IFLOW_PATH = '"C:\\Users\\霍冠华\\AppData\\Roaming\\npm\\node_modules\\@iflow-ai\\iflow-cli\\bundle\\iflow.js"';

// 中间件
// 配置CORS，允许所有域名访问（微信小程序需要）
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// 处理OPTIONS请求
app.options('*', (req, res) => {
  res.sendStatus(200);
});

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// 健康检查路由
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'iFlow API Server is running',
    timestamp: new Date().toISOString()
  });
});

// 执行iFlow命令的通用函数
function executeIflowCommand(command) {
  try {
    const fullCommand = `${NODE_PATH} ${IFLOW_PATH} ${command}`;
    const result = execSync(fullCommand, { encoding: 'utf-8', timeout: 30000 });
    return { success: true, output: result };
  } catch (error) {
    return { success: false, error: error.message, output: error.stdout || '' };
  }
}

// 获取已安装的智能体列表
app.get('/agents', (req, res) => {
  const result = executeIflowCommand('agent list');
  res.json(result);
});

// 获取已安装的MCP列表
app.get('/mcps', (req, res) => {
  const result = executeIflowCommand('mcp list');
  res.json(result);
});

// 执行iFlow命令（非交互式）
app.post('/execute', (req, res) => {
  const { prompt } = req.body;
  if (!prompt) {
    return res.status(400).json({ success: false, error: 'Prompt is required' });
  }
  
  const result = executeIflowCommand(`-p "${prompt}"`);
  res.json(result);
});

// 执行特定智能体
app.post('/agent/execute/:name', (req, res) => {
  const { name } = req.params;
  const { prompt } = req.body;
  if (!prompt) {
    return res.status(400).json({ success: false, error: 'Prompt is required' });
  }
  
  // 注意：需要根据iFlow的实际语法调整命令
  const result = executeIflowCommand(`-p "使用${name}智能体处理：${prompt}"`);
  res.json(result);
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`🚀 iFlow API Server running at http://localhost:${PORT}`);
  console.log(`📋 Available endpoints:`);
  console.log(`   GET  /health        - 健康检查`);
  console.log(`   GET  /agents        - 获取智能体列表`);
  console.log(`   GET  /mcps          - 获取MCP列表`);
  console.log(`   POST /execute       - 执行iFlow命令`);
  console.log(`   POST /agent/execute/:name - 执行特定智能体`);
});
