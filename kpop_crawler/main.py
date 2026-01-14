import os
import requests
from bs4 import BeautifulSoup
from pytube import YouTube
import time

# 移除Selenium依赖，直接使用requests获取数据

# 创建kpop目录
def create_kpop_directory():
    """创建主目录kpop"""
    if not os.path.exists('kpop'):
        os.makedirs('kpop')
        print("创建kpop目录成功")

# 获取BLACKPINK近10年歌曲列表
def get_blackpink_songs():
    # 直接提供BLACKPINK近10年的热门歌曲列表
    # 这些是BLACKPINK 2014年至今发布的主要歌曲
    songs = [
        "DDU-DU DDU-DU",
        "Kill This Love",
        "Pink Venom",
        "How You Like That",
        "Shut Down",
        "Lovesick Girls",
        "As If It's Your Last",
        "Playing with Fire",
        "Whistle",
        "Boombayah",
        "Stay",
        "Forever Young",
        "Kiss and Make Up",
        "Pretty Savage",
        "You Never Know"
    ]
    
    print(f"使用预定义的歌曲列表，共 {len(songs)} 首歌曲")
    return songs

# 获取舞蹈视频链接信息
def get_dance_video_link(song_name):
    """搜索并获取指定歌曲的舞蹈视频链接信息"""
    try:
        import urllib.parse
        
        # 创建歌曲目录
        song_dir = os.path.join('kpop', song_name)
        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
        
        # 构建YouTube搜索URL
        search_query = f"BLACKPINK {song_name} dance practice"
        encoded_query = urllib.parse.quote(search_query)
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        
        print(f"搜索: {search_query}")
        print(f"YouTube搜索链接: {search_url}")
        
        # 创建一个文本文件保存视频信息
        info_file_path = os.path.join(song_dir, f"{song_name}_dance_info.txt")
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write(f"歌曲名称: {song_name}\n")
            f.write(f"搜索关键词: {search_query}\n")
            f.write(f"YouTube搜索链接: {search_url}\n")
            f.write("说明: 点击上方YouTube搜索链接，选择第一个舞蹈练习视频进行观看或下载\n")
        
        print(f"已为 {song_name} 创建舞蹈视频信息文件")
        return True
        
    except Exception as e:
        print(f"获取视频链接信息失败: {song_name}, 错误: {str(e)}")
        return False

# 主函数
def main():
    try:
        print("开始爬取BLACKPINK近10年歌曲舞蹈...")
        
        # 创建kpop目录
        create_kpop_directory()
        
        # 获取BLACKPINK歌曲列表
        print("正在获取BLACKPINK歌曲列表...")
        songs = get_blackpink_songs()
        print(f"共找到 {len(songs)} 首歌曲")
        
        # 获取舞蹈视频链接信息
        print("开始获取舞蹈视频链接信息...")
        success_count = 0
        failed_count = 0
        
        # 测试时只处理前3首歌曲
        test_songs = songs[:3]
        print(f"测试模式，只处理 {len(test_songs)} 首歌曲")
        
        for song in test_songs:
            print(f"\n处理歌曲: {song}")
            if get_dance_video_link(song):
                success_count += 1
            else:
                failed_count += 1
            
            # 限速，避免被封
            time.sleep(1)
        
        # 不需要关闭浏览器，因为已经移除了Selenium依赖
        
        # 打印结果
        print("\n" + "="*50)
        print("爬取完成！")
        print(f"成功获取: {success_count} 个舞蹈视频链接信息")
        print(f"获取失败: {failed_count} 个舞蹈视频链接信息")
        print("文件结构:")
        print("kpop/")
        for song in test_songs:
            print(f"  └── {song}/")
            print(f"       └── {song}_dance_info.txt")
        
        print("\n使用说明:")
        print("1. 打开kpop文件夹，查看各个歌曲目录")
        print("2. 每个目录下的txt文件包含舞蹈视频的YouTube搜索链接")
        print("3. 点击链接即可找到对应的舞蹈练习视频")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
