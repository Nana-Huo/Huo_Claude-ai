// pages/booking/booking.js
const app = getApp()

Page({
  data: {
    service: null,
    selectedDate: '',
    selectedTime: '',
    selectedArtist: '',
    selectedNail: '',
    notes: '',
    bookingData: {
      serviceId: '',
      date: '',
      time: '',
      artistId: '',
      nailStyle: '',
      notes: '',
      contactName: '',
      contactPhone: '',
      totalPrice: 0
    },
    dates: [],
    timeSlots: [],
    nailArtists: [],
    nailStyles: [
      { id: 1, name: '方形', icon: '⬜' },
      { id: 2, name: '圆形', icon: '⭕' },
      { id: 3, name: '椭圆形', icon: '🥚' },
      { id: 4, name: '杏仁形', icon: '🌰' },
      { id: 5, name: '梯形', icon: '🔻' },
      { id: 6, name: '尖形', icon: '🔺' }
    ],
    availableNailStyles: [],
    contactInfo: {
      name: '',
      phone: ''
    },
    step: 1, // 1: 选择服务, 2: 选择时间, 3: 选择美甲师, 4: 选择甲型, 5: 确认信息
    maxStep: 5
  },

  onLoad(options) {
    const serviceId = options.serviceId
    this.loadServiceDetail(serviceId)
    this.loadDates()
    this.loadTimeSlots()
    this.loadNailArtists()
  },

  // 加载服务详情
  loadServiceDetail(serviceId) {
    const services = app.globalData.services || []
    const service = services.find(s => s.id == serviceId)
    
    if (service) {
      this.setData({
        service: service,
        'bookingData.serviceId': serviceId,
        'bookingData.totalPrice': service.price
      })
    } else {
      // 如果没有找到服务，使用默认数据
      const defaultService = {
        id: serviceId,
        name: '美甲服务',
        price: 88,
        duration: '60分钟'
      }
      this.setData({
        service: defaultService,
        'bookingData.serviceId': serviceId,
        'bookingData.totalPrice': defaultService.price
      })
    }
  },

  // 加载可选日期（未来7天）
  loadDates() {
    const dates = []
    const today = new Date()
    
    for (let i = 1; i <= 7; i++) {
      const date = new Date(today)
      date.setDate(today.getDate() + i)
      
      dates.push({
        date: this.formatDate(date),
        display: this.formatDisplayDate(date),
        disabled: false
      })
    }
    
    this.setData({ dates })
  },

  // 加载时间段
  loadTimeSlots() {
    const timeSlots = [
      '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
      '12:00', '12:30', '13:00', '13:30', '14:00', '14:30',
      '15:00', '15:30', '16:00', '16:30', '17:00', '17:30',
      '18:00', '18:30', '19:00', '19:30', '20:00'
    ]
    
    this.setData({ timeSlots })
  },

  // 加载美甲师列表
  loadNailArtists() {
    const nailArtists = [
      {
        id: 1,
        name: '李美甲师',
        avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b780?w=100',
        rating: 4.9,
        experience: '5年经验',
        specialty: '艺术美甲',
        available: true
      },
      {
        id: 2,
        name: '王美甲师',
        avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=100',
        rating: 4.8,
        experience: '3年经验',
        specialty: '基础美甲',
        available: true
      },
      {
        id: 3,
        name: '张美甲师',
        avatar: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=100',
        rating: 5.0,
        experience: '7年经验',
        specialty: '美甲护理',
        available: false
      },
      {
        id: 4,
        name: '陈美甲师',
        avatar: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=100',
        rating: 4.7,
        experience: '4年经验',
        specialty: '法式美甲',
        available: true
      }
    ]
    
    this.setData({ nailArtists })
  },

  // 格式化日期
  formatDate(date) {
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  },

  // 格式化显示日期
  formatDisplayDate(date) {
    const today = new Date()
    const tomorrow = new Date(today)
    tomorrow.setDate(today.getDate() + 1)
    
    if (date.toDateString() === today.toDateString()) {
      return '今天'
    } else if (date.toDateString() === tomorrow.toDateString()) {
      return '明天'
    } else {
      const weekdays = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
      return `${date.getMonth() + 1}月${date.getDate()}日 ${weekdays[date.getDay()]}`
    }
  },

  // 选择日期
  selectDate(e) {
    const date = e.currentTarget.dataset.date
    this.setData({
      selectedDate: date,
      'bookingData.date': date
    })
  },

  // 选择时间
  selectTime(e) {
    const time = e.currentTarget.dataset.time
    this.setData({
      selectedTime: time,
      'bookingData.time': time
    })
  },

  // 选择美甲师
  selectArtist(e) {
    const artistId = e.currentTarget.dataset.artistId
    const artist = this.data.nailArtists.find(a => a.id == artistId)
    
    if (artist && artist.available) {
      this.setData({
        selectedArtist: artistId,
        'bookingData.artistId': artistId
      })
    } else {
      wx.showToast({
        title: '该美甲师当前不可预约',
        icon: 'none'
      })
    }
  },

  // 选择甲型
  selectNail(e) {
    const nailId = e.currentTarget.dataset.nailId
    this.setData({
      selectedNail: nailId,
      'bookingData.nailStyle': nailId
    })
  },

  // 输入备注
  inputNotes(e) {
    const notes = e.detail.value
    this.setData({
      notes: notes,
      'bookingData.notes': notes
    })
  },

  // 输入联系信息
  inputContactInfo(e) {
    const field = e.currentTarget.dataset.field
    const value = e.detail.value
    this.setData({
      [`contactInfo.${field}`]: value,
      [`bookingData.contact${field.charAt(0).toUpperCase() + field.slice(1)}`]: value
    })
  },

  // 下一步
  nextStep() {
    if (this.data.step < this.data.maxStep) {
      // 验证当前步骤
      if (this.validateCurrentStep()) {
        this.setData({
          step: this.data.step + 1
        })
        
        // 如果进入最后一步，加载可用的甲型样式
        if (this.data.step === 4) {
          this.loadAvailableNailStyles()
        }
      }
    }
  },

  // 上一步
  prevStep() {
    if (this.data.step > 1) {
      this.setData({
        step: this.data.step - 1
      })
    }
  },

  // 验证当前步骤
  validateCurrentStep() {
    switch (this.data.step) {
      case 1:
        if (!this.data.selectedDate || !this.data.selectedTime) {
          wx.showToast({
            title: '请选择预约日期和时间',
            icon: 'none'
          })
          return false
        }
        break
      case 2:
        if (!this.data.selectedArtist) {
          wx.showToast({
            title: '请选择美甲师',
            icon: 'none'
          })
          return false
        }
        break
      case 3:
        if (!this.data.selectedNail) {
          wx.showToast({
            title: '请选择甲型',
            icon: 'none'
          })
          return false
        }
        break
      case 4:
        if (!this.data.contactInfo.name || !this.data.contactInfo.phone) {
          wx.showToast({
            title: '请填写联系信息',
            icon: 'none'
          })
          return false
        }
        if (!this.validatePhone(this.data.contactInfo.phone)) {
          wx.showToast({
            title: '请填写正确的手机号',
            icon: 'none'
          })
          return false
        }
        break
    }
    return true
  },

  // 验证手机号
  validatePhone(phone) {
    const phoneReg = /^1[3-9]\d{9}$/
    return phoneReg.test(phone)
  },

  // 加载可用的甲型样式
  loadAvailableNailStyles() {
    // 根据服务类型返回可用的甲型样式
    const serviceType = this.data.service.category
    let availableStyles = []
    
    if (serviceType === 1) {
      // 基础美甲：所有样式都可用
      availableStyles = this.data.nailStyles
    } else if (serviceType === 2) {
      // 艺术美甲：推荐复杂样式
      availableStyles = this.data.nailStyles.filter(style => 
        [2, 3, 4, 6].includes(style.id) // 圆形、椭圆形、杏仁形、尖形
      )
    } else {
      // 其他服务：推荐简单样式
      availableStyles = this.data.nailStyles.filter(style => 
        [1, 2, 3].includes(style.id) // 方形、圆形、椭圆形
      )
    }
    
    this.setData({
      availableNailStyles: availableStyles
    })
  },

  // 确认预约
  confirmBooking() {
    if (!this.validateCurrentStep()) {
      return
    }

    wx.showModal({
      title: '确认预约',
      content: `确认预约 ${this.data.service.name}？\n价格：¥${this.data.service.price}\n时间：${this.data.bookingData.date} ${this.data.bookingData.time}`,
      success: (res) => {
        if (res.confirm) {
          this.submitBooking()
        }
      }
    })
  },

  // 提交预约
  submitBooking() {
    wx.showLoading({
      title: '提交中...'
    })

    // 模拟API调用
    setTimeout(() => {
      wx.hideLoading()
      
      // 生成预约单号
      const bookingNo = 'NG' + Date.now()
      
      // 保存到本地存储
      const bookings = wx.getStorageSync('bookings') || []
      bookings.push({
        ...this.data.bookingData,
        bookingNo,
        status: 'pending',
        createTime: new Date().toISOString(),
        service: this.data.service
      })
      wx.setStorageSync('bookings', bookings)
      
      wx.showToast({
        title: '预约成功',
        icon: 'success'
      })

      // 跳转到确认页面
      setTimeout(() => {
        wx.redirectTo({
          url: `/pages/confirm/confirm?bookingNo=${bookingNo}`
        })
      }, 1500)
    }, 2000)
  },

  // 取消预约
  cancelBooking() {
    wx.showModal({
      title: '取消预约',
      content: '确定要取消预约吗？',
      success: (res) => {
        if (res.confirm) {
          wx.navigateBack()
        }
      }
    })
  }
})