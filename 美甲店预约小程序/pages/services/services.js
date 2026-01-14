// pages/services/services.js
const app = getApp()

Page({
  data: {
    categories: [
      { id: 0, name: '全部', active: true },
      { id: 1, name: '基础美甲', active: false },
      { id: 2, name: '艺术美甲', active: false },
      { id: 3, name: '美甲护理', active: false },
      { id: 4, name: '法式美甲', active: false },
      { id: 5, name: '美甲修复', active: false }
    ],
    services: [],
    filteredServices: [],
    selectedCategory: 0,
    loading: false
  },

  onLoad() {
    this.loadServices()
  },

  onShow() {
    // 页面显示时刷新数据
  },

  // 加载服务列表
  loadServices() {
    this.setData({ loading: true })
    
    // 模拟API调用
    setTimeout(() => {
      const allServices = [
        {
          id: 1,
          name: '基础美甲',
          price: 58,
          originalPrice: 68,
          image: 'https://images.unsplash.com/photo-1604654894610-df63bc536371?w=300',
          duration: '60分钟',
          category: 1,
          rating: 4.8,
          description: '基础美甲护理，修剪指甲形状，清洁指甲周围皮肤',
          features: ['指甲修剪', '形状打磨', '基础护理', '护甲油'],
          popular: true
        },
        {
          id: 2,
          name: '艺术美甲',
          price: 128,
          originalPrice: 158,
          image: 'https://images.unsplash.com/photo-1596462502278-27bfdc536348?w=300',
          duration: '90分钟',
          category: 2,
          rating: 4.9,
          description: '专业艺术美甲，个性化设计，创意图案绘制',
          features: ['个性化设计', '创意图案', '专业工具', '持久定型'],
          popular: true
        },
        {
          id: 3,
          name: '美甲护理',
          price: 88,
          originalPrice: 98,
          image: 'https://images.unsplash.com/photo-1582095133179-bfd08e2fc6b3?w=300',
          duration: '75分钟',
          category: 3,
          rating: 4.7,
          description: '深度美甲护理，修护受损指甲，强韧指甲结构',
          features: ['深度清洁', '指甲修复', '营养护理', '强韧处理'],
          popular: false
        },
        {
          id: 4,
          name: '法式美甲',
          price: 98,
          originalPrice: 118,
          image: 'https://images.unsplash.com/photo-1604654894610-df63bc536371?w=300',
          duration: '80分钟',
          category: 4,
          rating: 4.8,
          description: '经典法式美甲，优雅白色指尖，精致边缘处理',
          features: ['经典法式', '白色尖端', '精致边缘', '持久光泽'],
          popular: true
        },
        {
          id: 5,
          name: '美甲修复',
          price: 68,
          originalPrice: 78,
          image: 'https://images.unsplash.com/photo-1596462502278-27bfdc536348?w=300',
          duration: '45分钟',
          category: 5,
          rating: 4.6,
          description: '快速美甲修复，解决指甲断裂、破损问题',
          features: ['断裂修复', '破损填补', '快速处理', '即时修复'],
          popular: false
        },
        {
          id: 6,
          name: '水晶美甲',
          price: 158,
          originalPrice: 188,
          image: 'https://images.unsplash.com/photo-1582095133179-bfd08e2fc6b3?w=300',
          duration: '120分钟',
          category: 2,
          rating: 4.9,
          description: '高端水晶美甲，持久耐用，质感光滑',
          features: ['水晶材质', '持久耐用', '质感光滑', '高端工艺'],
          popular: true
        }
      ]
      
      this.setData({
        services: allServices,
        filteredServices: allServices,
        loading: false
      })
      
      app.globalData.services = allServices
    }, 500)
  },

  // 选择分类
  selectCategory(e) {
    const categoryId = e.currentTarget.dataset.categoryId
    const categories = this.data.categories.map(item => ({
      ...item,
      active: item.id === categoryId
    }))
    
    this.setData({
      categories,
      selectedCategory: categoryId
    })
    
    this.filterServices(categoryId)
  },

  // 筛选服务
  filterServices(categoryId) {
    let filteredServices
    if (categoryId === 0) {
      // 全部服务
      filteredServices = this.data.services
    } else {
      // 按分类筛选
      filteredServices = this.data.services.filter(service => service.category === categoryId)
    }
    
    this.setData({
      filteredServices
    })
  },

  // 跳转到预约页面
  goToBooking(e) {
    const serviceId = e.currentTarget.dataset.serviceId
    wx.navigateTo({
      url: `/pages/booking/booking?serviceId=${serviceId}`
    })
  },

  // 跳转到服务详情
  goToServiceDetail(e) {
    const serviceId = e.currentTarget.dataset.serviceId
    wx.navigateTo({
      url: `/pages/service-detail/service-detail?serviceId=${serviceId}`
    })
  },

  // 搜索服务
  onSearch(e) {
    const keyword = e.detail.value
    if (!keyword.trim()) {
      this.setData({
        filteredServices: this.data.services
      })
      return
    }
    
    const filteredServices = this.data.services.filter(service => 
      service.name.includes(keyword) || 
      service.description.includes(keyword)
    )
    
    this.setData({
      filteredServices
    })
  },

  // 下拉刷新
  onPullDownRefresh() {
    this.loadServices()
    setTimeout(() => {
      wx.stopPullDownRefresh()
    }, 1000)
  }
})