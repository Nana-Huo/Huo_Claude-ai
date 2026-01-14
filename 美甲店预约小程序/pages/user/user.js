// pages/user/user.js
Page({
  data: {
    userInfo: null,
    isLoggedIn: false,
    userStats: {
      totalBookings: 0,
      completedBookings: 0,
      cancelledBookings: 0,
      points: 0
    },
    menuItems: [
      {
        id: 'my-bookings',
        title: '我的预约',
        icon: '📅',
        description: '查看预约记录',
        action: 'goToBookings'
      },
      {
        id: 'favorites',
        title: '收藏的服务',
        icon: '❤️',
        description: '我的服务收藏',
        action: 'goToFavorites'
      },
      {
        id: 'coupons',
        title: '优惠券',
        icon: '🎫',
        description: '可用优惠券',
        action: 'goToCoupons'
      },
      {
        id: 'points',
        title: '积分商城',
        icon: '⭐',
        description: '积分兑换',
        action: 'goToPoints'
      },
      {
        id: 'addresses',
        title: '收货地址',
        icon: '📍',
        description: '管理地址',
        action: 'goToAddresses'
      },
      {
        id: 'feedback',
        title: '意见反馈',
        icon: '💬',
        description: '提出建议',
        action: 'goToFeedback'
      },
      {
        id: 'customer-service',
        title: '客服中心',
        icon: '🎧',
        description: '在线客服',
        action: 'goToCustomerService'
      },
      {
        id: 'settings',
        title: '设置',
        icon: '⚙️',
        description: '账户设置',
        action: 'goToSettings'
      }
    ],
    quickActions: [
      {
        id: 'book-now',
        title: '立即预约',
        icon: '✨',
        action: 'quickBook'
      },
      {
        id: 'contact-us',
        title: '联系我们',
        icon: '📞',
        action: 'contactUs'
      }
    ]
  },

  onLoad() {
    this.checkLoginStatus()
    this.loadUserData()
  },

  onShow() {
    this.loadUserData()
  },

  // 检查登录状态
  checkLoginStatus() {
    const userInfo = wx.getStorageSync('userInfo')
    if (userInfo) {
      this.setData({
        userInfo: userInfo,
        isLoggedIn: true
      })
    }
  },

  // 加载用户数据
  loadUserData() {
    if (!this.data.isLoggedIn) {
      return
    }

    // 从本地存储加载预约数据
    const bookings = wx.getStorageSync('bookings') || []
    const userStats = {
      totalBookings: bookings.length,
      completedBookings: bookings.filter(b => b.status === 'completed').length,
      cancelledBookings: bookings.filter(b => b.status === 'cancelled').length,
      points: Math.floor(bookings.filter(b => b.status === 'completed').length * 10) // 每完成一个预约获得10积分
    }

    this.setData({
      userStats
    })
  },

  // 登录
  login() {
    const that = this
    wx.getUserProfile({
      desc: '用于完善用户资料',
      success: (res) => {
        const userInfo = res.userInfo
        wx.setStorageSync('userInfo', userInfo)
        
        that.setData({
          userInfo: userInfo,
          isLoggedIn: true
        })

        wx.showToast({
          title: '登录成功',
          icon: 'success'
        })

        // 登录后刷新数据
        that.loadUserData()
      },
      fail: (err) => {
        wx.showToast({
          title: '登录失败',
          icon: 'none'
        })
      }
    })
  },

  // 退出登录
  logout() {
    wx.showModal({
      title: '退出登录',
      content: '确定要退出登录吗？',
      success: (res) => {
        if (res.confirm) {
          wx.removeStorageSync('userInfo')
          this.setData({
            userInfo: null,
            isLoggedIn: false,
            userStats: {
              totalBookings: 0,
              completedBookings: 0,
              cancelledBookings: 0,
              points: 0
            }
          })
          
          wx.showToast({
            title: '已退出登录',
            icon: 'success'
          })
        }
      }
    })
  },

  // 菜单项点击
  onMenuItemTap(e) {
    const action = e.currentTarget.dataset.action
    if (this[action]) {
      this[action]()
    }
  },

  // 快速操作点击
  onQuickActionTap(e) {
    const action = e.currentTarget.dataset.action
    if (this[action]) {
      this[action]()
    }
  },

  // 跳转到预约列表
  goToBookings() {
    wx.switchTab({
      url: '/pages/orders/orders'
    })
  },

  // 跳转到收藏
  goToFavorites() {
    wx.navigateTo({
      url: '/pages/favorites/favorites'
    })
  },

  // 跳转到优惠券
  goToCoupons() {
    wx.navigateTo({
      url: '/pages/coupons/coupons'
    })
  },

  // 跳转到积分商城
  goToPoints() {
    wx.navigateTo({
      url: '/pages/points/points'
    })
  },

  // 跳转到地址管理
  goToAddresses() {
    wx.navigateTo({
      url: '/pages/addresses/addresses'
    })
  },

  // 跳转到反馈
  goToFeedback() {
    wx.navigateTo({
      url: '/pages/feedback/feedback'
    })
  },

  // 跳转到客服
  goToCustomerService() {
    wx.makePhoneCall({
      phoneNumber: '400-123-4567'
    })
  },

  // 跳转到设置
  goToSettings() {
    wx.navigateTo({
      url: '/pages/settings/settings'
    })
  },

  // 快速预约
  quickBook() {
    wx.switchTab({
      url: '/pages/index/index'
    })
  },

  // 联系我们
  contactUs() {
    wx.showActionSheet({
      itemList: ['拨打电话', '在线客服'],
      success: (res) => {
        if (res.tapIndex === 0) {
          wx.makePhoneCall({
            phoneNumber: '400-123-4567'
          })
        } else if (res.tapIndex === 1) {
          wx.navigateTo({
            url: '/pages/customer-service/customer-service'
          })
        }
      }
    })
  },

  // 分享小程序
  share() {
    wx.showShareMenu({
      withShareTicket: true
    })
  },

  // 更新用户信息
  updateUserInfo() {
    wx.navigateTo({
      url: '/pages/edit-profile/edit-profile'
    })
  },

  // 查看会员等级
  viewMembership() {
    wx.navigateTo({
      url: '/pages/membership/membership'
    })
  },

  // 查看积分明细
  viewPointsDetail() {
    wx.navigateTo({
      url: '/pages/points-detail/points-detail'
    })
  },

  // 编辑用户信息
  editProfile() {
    wx.navigateTo({
      url: '/pages/edit-profile/edit-profile'
    })
  }
})