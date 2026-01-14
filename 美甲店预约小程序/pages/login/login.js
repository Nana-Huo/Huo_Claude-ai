// pages/login/login.js
Page({
  data: {
    phone: '',
    code: '',
    gettingCode: false,
    countdown: 60,
    codeTimer: null,
    agreeTerms: false
  },

  onLoad(options) {
    // 检查是否已经登录
    const userInfo = wx.getStorageSync('userInfo')
    if (userInfo) {
      wx.switchTab({
        url: '/pages/index/index'
      })
      return
    }

    // 如果有回调地址，保存下来
    if (options.redirect) {
      this.setData({
        redirectUrl: options.redirect
      })
    }
  },

  onUnload() {
    // 清理定时器
    if (this.data.codeTimer) {
      clearInterval(this.data.codeTimer)
    }
  },

  // 输入手机号
  onPhoneInput(e) {
    this.setData({
      phone: e.detail.value
    })
  },

  // 输入验证码
  onCodeInput(e) {
    this.setData({
      code: e.detail.value
    })
  },

  // 切换同意条款
  onTermsChange(e) {
    this.setData({
      agreeTerms: e.detail.value
    })
  },

  // 获取验证码
  onGetCode() {
    const phone = this.data.phone.trim()
    
    if (!phone) {
      wx.showToast({
        title: '请输入手机号',
        icon: 'none'
      })
      return
    }

    if (!/^1[3-9]\d{9}$/.test(phone)) {
      wx.showToast({
        title: '手机号格式不正确',
        icon: 'none'
      })
      return
    }

    if (this.data.gettingCode) {
      return
    }

    this.setData({
      gettingCode: true
    })

    // 模拟发送验证码
    wx.showLoading({
      title: '发送中...',
      mask: true
    })

    setTimeout(() => {
      wx.hideLoading()
      
      wx.showToast({
        title: '验证码已发送',
        icon: 'success'
      })

      this.startCountdown()
      this.setData({
        gettingCode: false
      })
    }, 1500)
  },

  // 开始倒计时
  startCountdown() {
    let countdown = 60
    const timer = setInterval(() => {
      countdown--
      this.setData({
        countdown: countdown
      })

      if (countdown <= 0) {
        clearInterval(timer)
        this.setData({
          countdown: 60,
          codeTimer: null
        })
      }
    }, 1000)

    this.setData({
      codeTimer: timer
    })
  },

  // 登录
  onLogin() {
    const phone = this.data.phone.trim()
    const code = this.data.code.trim()

    // 验证输入
    if (!phone) {
      wx.showToast({
        title: '请输入手机号',
        icon: 'none'
      })
      return
    }

    if (!/^1[3-9]\d{9}$/.test(phone)) {
      wx.showToast({
        title: '手机号格式不正确',
        icon: 'none'
      })
      return
    }

    if (!code) {
      wx.showToast({
        title: '请输入验证码',
        icon: 'none'
      })
      return
    }

    if (code.length !== 6) {
      wx.showToast({
        title: '验证码格式不正确',
        icon: 'none'
      })
      return
    }

    if (!this.data.agreeTerms) {
      wx.showToast({
        title: '请同意用户协议和隐私政策',
        icon: 'none'
      })
      return
    }

    // 模拟登录
    wx.showLoading({
      title: '登录中...',
      mask: true
    })

    setTimeout(() => {
      wx.hideLoading()

      // 模拟用户数据
      const userInfo = {
        id: Date.now(),
        phone: phone,
        nickName: `用户${phone.slice(-4)}`,
        avatarUrl: '',
        loginTime: new Date().toISOString()
      }

      // 保存用户信息
      wx.setStorageSync('userInfo', userInfo)

      wx.showToast({
        title: '登录成功',
        icon: 'success'
      })

      // 跳转到指定页面或首页
      setTimeout(() => {
        if (this.data.redirectUrl) {
          wx.redirectTo({
            url: this.data.redirectUrl
          })
        } else {
          wx.switchTab({
            url: '/pages/index/index'
          })
        }
      }, 1500)
    }, 2000)
  },

  // 微信授权登录
  onWechatLogin() {
    const that = this
    wx.getUserProfile({
      desc: '用于完善用户资料',
      success: (res) => {
        const userInfo = res.userInfo
        
        // 保存用户信息
        wx.setStorageSync('userInfo', userInfo)

        wx.showToast({
          title: '登录成功',
          icon: 'success'
        })

        // 跳转到指定页面或首页
        setTimeout(() => {
          if (that.data.redirectUrl) {
            wx.redirectTo({
              url: that.data.redirectUrl
            })
          } else {
            wx.switchTab({
              url: '/pages/index/index'
            })
          }
        }, 1500)
      },
      fail: (err) => {
        wx.showToast({
          title: '登录失败',
          icon: 'none'
        })
      }
    })
  },

  // 查看用户协议
  onViewTerms() {
    wx.navigateTo({
      url: '/pages/terms/terms'
    })
  },

  // 查看隐私政策
  onViewPrivacy() {
    wx.navigateTo({
      url: '/pages/privacy/privacy'
    })
  },

  // 游客模式
  onGuestMode() {
    wx.showModal({
      title: '游客模式',
      content: '游客模式下部分功能可能受限，确定要继续吗？',
      success: (res) => {
        if (res.confirm) {
          const guestInfo = {
            id: Date.now(),
            phone: '',
            nickName: '游客',
            avatarUrl: '',
            isGuest: true,
            loginTime: new Date().toISOString()
          }

          wx.setStorageSync('userInfo', guestInfo)

          wx.switchTab({
            url: '/pages/index/index'
          })
        }
      }
    })
  },

  // 跳转到首页
  goToHome() {
    wx.switchTab({
      url: '/pages/index/index'
    })
  }
})