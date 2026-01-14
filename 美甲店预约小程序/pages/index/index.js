// pages/index/index.js
const app = getApp()

Page({
  data: {
    bannerList: [
      {
        id: 1,
        image: 'https://images.unsplash.com/photo-1604654894610-df63bc536371?w=750',
        title: '精美美甲服务'
      },
      {
        id: 2,
        image: 'https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=750',
        title: '专业美甲师'
      },
      {
        id: 3,
        image: 'https://images.unsplash.com/photo-1582095133179-bfd08e2fc6b3?w=750',
        title: '预约便捷服务'
      }
    ],
    services: [
      {
        id: 1,
        name: '基础美甲',
        price: 58,
        image: 'https://images.unsplash.com/photo-1604654894610-df63bc536371?w=300',
        duration: '60分钟'
      },
      {
        id: 2,
        name: '艺术美甲',
        price: 128,
        image: 'https://images.unsplash.com/photo-1596462502278-27bfdc536348?w=300',
        duration: '90分钟'
      },
      {
        id: 3,
        name: '美甲护理',
        price: 88,
        image: 'https://images.unsplash.com/photo-1582095133179-bfd08e2fc6b3?w=300',
        duration: '75分钟'
      },
      {
        id: 4,
        name: '法式美甲',
        price: 98,
        image: 'https://images.unsplash.com/photo-1604654894610-df63bc536371?w=300',
        duration: '80分钟'
      }
    ],
    nailArtists: [
      {
        id: 1,
        name: '李美甲师',
        avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b780?w=100',
        rating: 4.9,
        experience: '5年经验',
        specialty: '艺术美甲'
      },
      {
        id: 2,
        name: '王美甲师',
        avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=100',
        rating: 4.8,
        experience: '3年经验',
        specialty: '基础美甲'
      },
      {
        id: 3,
        name: '张美甲师',
        avatar: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=100',
        rating: 5.0,
        experience: '7年经验',
        specialty: '美甲护理'
      }
    ],
    currentBanner: 0
  },

  onLoad() {
    this.loadServices()
    this.loadNailArtists()
    this.startBannerAutoplay()
  },

  onReady() {
    // 页面渲染完成
  },

  onShow() {
    // 页面显示
  },

  onHide() {
    // 页面隐藏
  },

  onUnload() {
    // 页面卸载
    this.stopBannerAutoplay()
  },

  // 加载服务列表
  loadServices() {
    // 模拟数据，实际项目中从API获取
    const services = this.data.services
    app.globalData.services = services
  },

  // 加载美甲师列表
  loadNailArtists() {
    // 模拟数据，实际项目中从API获取
    const nailArtists = this.data.nailArtists
    app.globalData.nailArtists = nailArtists
  },

  // 轮播图自动播放
  startBannerAutoplay() {
    this.bannerTimer = setInterval(() => {
      let current = this.data.currentBanner
      let bannerLength = this.data.bannerList.length
      current = (current + 1) % bannerLength
      this.setData({
        currentBanner: current
      })
    }, 3000)
  },

  // 停止轮播图自动播放
  stopBannerAutoplay() {
    if (this.bannerTimer) {
      clearInterval(this.bannerTimer)
    }
  },

  // 轮播图变化
  onBannerChange(e) {
    this.setData({
      currentBanner: e.detail.current
    })
  },

  // 跳转到服务页面
  goToServices() {
    wx.switchTab({
      url: '/pages/services/services'
    })
  },

  // 跳转到预约页面
  goToBooking(e) {
    const serviceId = e.currentTarget.dataset.serviceId
    wx.navigateTo({
      url: `/pages/booking/booking?serviceId=${serviceId}`
    })
  },

  // 跳转到美甲师详情
  goToArtistDetail(e) {
    const artistId = e.currentTarget.dataset.artistId
    wx.navigateTo({
      url: `/pages/artist/artist?artistId=${artistId}`
    })
  },

  // 搜索功能
  onSearch() {
    wx.navigateTo({
      url: '/pages/search/search'
    })
  }
})