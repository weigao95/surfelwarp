import cv2
import os

# 定义视频的帧率和编码
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码

# 初始化 VideoWriter 对象（我们稍后会重新设置这个对象）
video = None
output_video_file = 'demo/breathe.mp4'

for i in range(1,301):
    image_path = f"/home/rvclab/dev/surfelwarp/test_data/dataset/breathe/frame-{i:06d}.color.png" 
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} does not exist and will be skipped.")
        continue
    
    # 读取图片s
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read {image_path}, skipping.")
        continue

    # 如果 video 对象未初始化（即这是第一帧）
    if video is None:
        height, width, layers = frame.shape
        video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # 写入帧到视频
    video.write(frame)

# 释放资源
if video is not None:
    video.release()
else:
    print("No videos were written. Check if the image paths are correct.")
