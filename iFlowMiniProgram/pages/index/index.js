// index.js
const app = getApp()

Page({
  data: {
    messages: [
      {
        id: Date.now(),
        text: '您好！我是iFlow AI助手，很高兴为您服务。您可以向我发送命令，或者点击智能体按钮选择特定功能的智能体。',
        isUser: false
      }
    ],
    inputText: '',
    toView: '',
    selectedAgent: null
  },

  onLoad() {
    // 页面加载时获取全局消息和智能体
    if (app.globalData.messages.length > 0) {
      this.setData({
        messages: app.globalData.messages
      })
    }
    
    if (app.globalData.selectedAgent) {
      this.setData({
        selectedAgent: app.globalData.selectedAgent
      })
    }
  },
  
  onShow() {
    // 页面显示时同步智能体选择状态
    if (app.globalData.selectedAgent) {
      this.setData({
        selectedAgent: app.globalData.selectedAgent
      })
    }
  },

  // 输入事件
  onInput(e) {
    this.setData({
      inputText: e.detail.value
    })
  },

  // 发送消息
  sendMessage() {
    if (!this.data.inputText.trim()) return

    // 添加用户消息
    const userMessage = {
      id: Date.now(),
      text: this.data.inputText,
      isUser: true
    }

    const messages = [...this.data.messages, userMessage]
    this.setData({
      messages: messages,
      inputText: '',
      toView: userMessage.id.toString()
    })

    // 保存到全局数据
    app.globalData.messages = messages

    // 发送请求到API服务
    this.sendRequest(this.data.inputText)
  },

  // 发送请求
  sendRequest(prompt) {
    wx.showLoading({
      title: '思考中...',
    })

    // 检查是否有选定的智能体
    if (app.globalData.selectedAgent) {
      // 使用特定智能体
      wx.request({
        url: `${app.globalData.baseURL}/agent/execute/${encodeURIComponent(app.globalData.selectedAgent.name)}`,
        method: 'POST',
        data: {
          prompt: prompt
        },
        success: (res) => {
          console.log('API Response:', res.data)
          this.handleResponse(res.data)
        },
        fail: (err) => {
          console.error('API Request Failed:', err)
          this.handleError(err)
        },
        complete: () => {
          wx.hideLoading()
        }
      })
    } else {
      // 使用默认模式
      wx.request({
        url: `${app.globalData.baseURL}/execute`,
        method: 'POST',
        data: {
          prompt: prompt
        },
        success: (res) => {
          console.log('API Response:', res.data)
          this.handleResponse(res.data)
        },
        fail: (err) => {
          console.error('API Request Failed:', err)
          this.handleError(err)
        },
        complete: () => {
          wx.hideLoading()
        }
      })
    }
  },

  // 处理响应
  handleResponse(response) {
    if (response.success && response.output) {
      const aiMessage = {
        id: Date.now(),
        text: response.output,
        isUser: false
      }

      const messages = [...this.data.messages, aiMessage]
      this.setData({
        messages: messages,
        toView: aiMessage.id.toString()
      })

      // 保存到全局数据
      app.globalData.messages = messages
    } else {
      this.handleError(response.error || '请求失败')
    }
  },

  // 处理错误
  handleError(error) {
    const errorMessage = {
      id: Date.now(),
      text: `错误: ${error}`,
      isUser: false
    }

    const messages = [...this.data.messages, errorMessage]
    this.setData({
      messages: messages,
      toView: errorMessage.id.toString()
    })

    // 保存到全局数据
    app.globalData.messages = messages
  },

  // 显示智能体列表
  showAgents() {
    wx.navigateTo({
      url: '/pages/agents/agents'
    })
  }
})
