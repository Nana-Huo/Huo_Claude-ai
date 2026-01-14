// pages/orders/orders.js
Page({
  data: {
    activeTab: 'all',
    orders: [],
    filteredOrders: [],
    tabs: [
      { id: 'all', name: '全部', count: 0 },
      { id: 'pending', name: '待确认', count: 0 },
      { id: 'confirmed', name: '已确认', count: 0 },
      { id: 'completed', name: '已完成', count: 0 },
      { id: 'cancelled', name: '已取消', count: 0 }
    ]
  },

  onLoad() {
    this.loadOrders()
  },

  onShow() {
    this.loadOrders()
  },

  // 加载订单数据
  loadOrders() {
    const orders = wx.getStorageSync('bookings') || []
    
    // 计算各个状态的订单数量
    const tabCounts = {
      all: orders.length,
      pending: orders.filter(o => o.status === 'pending').length,
      confirmed: orders.filter(o => o.status === 'confirmed').length,
      completed: orders.filter(o => o.status === 'completed').length,
      cancelled: orders.filter(o => o.status === 'cancelled').length
    }

    const updatedTabs = this.data.tabs.map(tab => ({
      ...tab,
      count: tabCounts[tab.id] || 0
    }))

    this.setData({
      orders,
      tabs: updatedTabs
    })

    this.filterOrders()
  },

  // 筛选订单
  filterOrders() {
    let filteredOrders = []
    
    if (this.data.activeTab === 'all') {
      filteredOrders = this.data.orders
    } else {
      filteredOrders = this.data.orders.filter(order => order.status === this.data.activeTab)
    }

    // 按时间倒序排列
    filteredOrders.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))

    this.setData({
      filteredOrders
    })
  },

  // 切换标签
  onTabChange(e) {
    const tabId = e.currentTarget.dataset.tab
    this.setData({
      activeTab: tabId
    })
    this.filterOrders()
  },

  // 跳转到订单详情
  onOrderTap(e) {
    const orderId = e.currentTarget.dataset.id
    wx.navigateTo({
      url: `/pages/confirm/confirm?id=${orderId}`
    })
  },

  // 取消订单
  onCancelOrder(e) {
    const orderId = e.currentTarget.dataset.id
    
    wx.showModal({
      title: '取消订单',
      content: '确定要取消这个预约吗？',
      success: (res) => {
        if (res.confirm) {
          this.updateOrderStatus(orderId, 'cancelled')
        }
      }
    })
  },

  // 确认订单
  onConfirmOrder(e) {
    const orderId = e.currentTarget.dataset.id
    
    wx.showModal({
      title: '确认订单',
      content: '确认已到店并完成服务吗？',
      success: (res) => {
        if (res.confirm) {
          this.updateOrderStatus(orderId, 'completed')
        }
      }
    })
  },

  // 更新订单状态
  updateOrderStatus(orderId, newStatus) {
    const orders = this.data.orders.map(order => {
      if (order.id === orderId) {
        return { ...order, status: newStatus }
      }
      return order
    })

    wx.setStorageSync('bookings', orders)
    
    wx.showToast({
      title: '操作成功',
      icon: 'success'
    })

    this.loadOrders()
  },

  // 再次预约
  onReBook(e) {
    const order = e.currentTarget.dataset.order
    wx.navigateTo({
      url: `/pages/booking/booking?serviceId=${order.serviceId}&artistId=${order.artistId}`
    })
  },

  // 获取状态显示文本
  getStatusText(status) {
    const statusMap = {
      'pending': '待确认',
      'confirmed': '已确认',
      'completed': '已完成',
      'cancelled': '已取消'
    }
    return statusMap[status] || status
  },

  // 获取状态样式
  getStatusClass(status) {
    const classMap = {
      'pending': 'status-pending',
      'confirmed': 'status-confirmed',
      'completed': 'status-completed',
      'cancelled': 'status-cancelled'
    }
    return classMap[status] || ''
  },

  // 跳转到首页预约
  goToBooking() {
    wx.switchTab({
      url: '/pages/index/index'
    })
  },

  // 下拉刷新
  onPullDownRefresh() {
    this.loadOrders()
    setTimeout(() => {
      wx.stopPullDownRefresh()
    }, 1000)
  }
})