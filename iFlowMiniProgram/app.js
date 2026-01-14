// app.js
App({
  globalData: {
    baseURL: 'http://localhost:4000',
    selectedAgent: null,
    messages: []
  },

  onLaunch() {
    // 初始化应用
    console.log('iFlow Mini Program launched');
    
    // 健康检查
    this.checkHealth();
  },

  // 健康检查
  checkHealth() {
    const that = this;
    wx.request({
      url: `${this.globalData.baseURL}/health`,
      method: 'GET',
      success(res) {
        console.log('API Server Health Check:', res.data);
        if (res.data.status === 'ok') {
          wx.showToast({
            title: '服务连接成功',
            icon: 'success',
            duration: 2000
          });
        }
      },
      fail(err) {
        console.error('API Server Health Check Failed:', err);
        wx.showToast({
          title: '服务连接失败',
          icon: 'error',
          duration: 2000
        });
      }
    });
  }
});