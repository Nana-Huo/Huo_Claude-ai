// agents.js
const app = getApp()

Page({
  data: {
    agents: [],
    loading: true
  },

  onLoad() {
    // 加载智能体列表
    this.loadAgents()
  },

  // 加载智能体列表
  loadAgents() {
    wx.request({
      url: `${app.globalData.baseURL}/agents`,
      method: 'GET',
      success: (res) => {
        console.log('Agents Response:', res.data)
        this.parseAgents(res.data)
      },
      fail: (err) => {
        console.error('Failed to load agents:', err)
        this.setData({
          loading: false
        })
        
        wx.showToast({
          title: '加载智能体失败',
          icon: 'error'
        })
      }
    })
  },

  // 解析智能体列表
  parseAgents(response) {
    if (response.success && response.output) {
      const output = response.output
      const agents = []
      const lines = output.split('\n')
      
      let currentAgent = null
      
      for (let line of lines) {
        line = line.trim()
        
        // 检查是否是智能体名称行
        if (line.startsWith('•') && line.includes('(')) {
          // 如果有当前智能体，先添加到列表
          if (currentAgent) {
            agents.push(currentAgent)
          }
          
          // 提取智能体名称
          const nameMatch = line.match(/•\s*([^()]+)\s*\([^)]+\)/)
          if (nameMatch && nameMatch[1]) {
            // 去除颜色代码
            const name = nameMatch[1].replace(/\u001b\[\d+m/g, '')
            currentAgent = {
              id: Date.now() + Math.random(),
              name: name,
              description: ''
            }
          }
        } 
        // 检查是否是描述行
        else if (currentAgent && line.includes('描述:')) {
          // 提取描述内容，忽略前面的空格
          const descriptionMatch = line.match(/描述:(.+)/)
          if (descriptionMatch && descriptionMatch[1]) {
            currentAgent.description = descriptionMatch[1].trim()
          }
        }
      }
      
      // 添加最后一个智能体
      if (currentAgent) {
        agents.push(currentAgent)
      }
      
      this.setData({
        agents: agents,
        loading: false
      })
    } else {
      this.setData({
        loading: false
      })
      
      wx.showToast({
        title: '解析智能体失败',
        icon: 'error'
      })
    }
  },

  // 选择智能体
  selectAgent(e) {
    const agent = e.currentTarget.dataset.agent
    
    // 保存到全局数据
    app.globalData.selectedAgent = agent
    
    // 返回上一页
    wx.navigateBack()
    
    // 显示选择结果
    wx.showToast({
      title: `已选择${agent.name}`,
      icon: 'success'
    })
  }
})
