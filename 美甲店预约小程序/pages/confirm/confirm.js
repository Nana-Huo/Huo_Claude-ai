// pages/confirm/confirm.js
Page({
  data: {
    bookingData: null,
    orderStatus: 'pending', // pending, confirmed, cancelled, completed
    countdown: 300, // 5分钟倒计时（秒）
    orderNumber: '',
    showSuccessModal: false
  },

  onLoad(options) {
    const bookingNo = options.bookingNo
    if (bookingNo) {
      this.loadBookingDetail(bookingNo)
    }
    this.startCountdown()
  },

  onUnload() {
    this.clearCountdown()
  },

  // 加载预约详情
  loadBookingDetail(bookingNo) {
    const bookings = wx.getStorageSync('bookings') || []
    const booking = bookings.find(b => b.bookingNo === bookingNo)
    
    if (booking) {
      this.setData({
        bookingData: booking,
        orderNumber: bookingNo
      })
    } else {
      wx.showToast({
        title: '预约信息不存在',
        icon: 'none'
      })
      setTimeout(() => {
        wx.navigateBack()
      }, 1500)
    }
  },

  // 开始倒计时
  startCountdown() {
    this.countdownTimer = setInterval(() => {
      let countdown = this.data.countdown
      if (countdown > 0) {
        countdown--
        this.setData({ countdown })
      } else {
        // 倒计时结束，自动取消订单
        this.cancelOrder()
      }
    }, 1000)
  },

  // 清除倒计时
  clearCountdown() {
    if (this.countdownTimer) {
      clearInterval(this.countdownTimer)
    }
  },

  // 格式化倒计时显示
  formatCountdown(seconds) {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
  },

  // 确认订单
  confirmOrder() {
    if (this.data.orderStatus !== 'pending') {
      return
    }

    wx.showModal({
      title: '确认预约',
      content: '确定要确认这个预约吗？',
      success: (res) => {
        if (res.confirm) {
          this.updateOrderStatus('confirmed')
        }
      }
    })
  },

  // 取消订单
  cancelOrder() {
    if (this.data.orderStatus !== 'pending') {
      return
    }

    wx.showModal({
      title: '取消预约',
      content: '确定要取消这个预约吗？',
      success: (res) => {
        if (res.confirm) {
          this.updateOrderStatus('cancelled')
        }
      }
    })
  },

  // 更新订单状态
  updateOrderStatus(status) {
    const bookings = wx.getStorageSync('bookings') || []
    const bookingIndex = bookings.findIndex(b => b.bookingNo === this.data.orderNumber)
    
    if (bookingIndex >= 0) {
      bookings[bookingIndex].status = status
      bookings[bookingIndex].updateTime = new Date().toISOString()
      wx.setStorageSync('bookings', bookings)
      
      this.setData({
        orderStatus: status,
        bookingData: bookings[bookingIndex]
      })

      if (status === 'confirmed') {
        this.showSuccessModal()
        this.clearCountdown()
      } else if (status === 'cancelled') {
        wx.showToast({
          title: '预约已取消',
          icon: 'success'
        })
        setTimeout(() => {
          wx.navigateBack()
        }, 1500)
      }
    }
  },

  // 显示成功模态框
  showSuccessModal() {
    this.setData({
      showSuccessModal: true
    })
  },

  // 隐藏成功模态框
  hideSuccessModal() {
    this.setData({
      showSuccessModal: false
    })
  },

  // 跳转到首页
  goToHome() {
    wx.switchTab({
      url: '/pages/index/index'
    })
  },

  // 跳转到订单列表
  goToOrders() {
    wx.switchTab({
      url: '/pages/orders/orders'
    })
  },

  // 联系客服
  contactService() {
    wx.makePhoneCall({
      phoneNumber: '400-123-4567' // 替换为实际客服电话
    })
  },

  // 分享预约
  shareBooking() {
    const that = this
    wx.showModal({
      title: '分享预约',
      content: '邀请朋友一起预约美甲服务',
      success: (res) => {
        if (res.confirm) {
          // 这里可以调用分享API
          wx.showToast({
            title: '分享功能开发中',
            icon: 'none'
          })
        }
      }
    })
  }
})