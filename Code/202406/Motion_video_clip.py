import cv2
import os
import numpy as np
import pandas as pd
import csv



def find_and_process_mkv_files(directory):
    """
    遍历文件夹及其子文件夹中的所有 .mkv 文件，并逐一进行处理。
    :param directory: str, 要遍历的根目录
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                process_mkv(file_path)

def compute_frame_differences(video_path):
    """
    计算视频的差分帧及其像素变化总值。
    
    :param video_path: str, 视频文件的路径
    :return: list, 每个差分帧的像素变化总值
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    diff_sums = []
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取结束
        
        # 将帧转换为灰度
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # 计算当前帧与上一帧的差分
            frame_diff = cv2.absdiff(gray_frame, prev_frame)
            
            # 计算差分帧的像素变化总值
            diff_sum = np.sum(frame_diff)
            diff_sums.append(diff_sum)
        
        # 更新上一帧
        prev_frame = gray_frame
    
    cap.release()
    return diff_sums

def smooth(D, k):
    # 对一维数组D进行平滑处理，返回平滑后的数组Z。
    # 数组D
    # 平滑窗口大小
    # 平滑后的数组Z，与D具有相同的长度
    """
    对一维数组D进行平滑处理，返回平滑后的数组Z。
    
    Args:
        D (np.ndarray): 一维数组，需要进行平滑处理的数组。
        k (int): 平滑窗口大小，必须为奇数。
    
    Returns:
        np.ndarray: 平滑后的数组Z，与D具有相同的长度。
    
    """
    # 数组D的长度
    size = len(D)
    # 初始化平滑后的数组Z，与D具有相同的长度，并全部赋值为0
    Z = np.zeros(size)
    # 计算平滑窗口的一半大小
    b = (k - 1) // 2
    # 遍历数组D的每个元素
    for i in range(size):
        # 计算当前元素对应平滑窗口的起始下标
        start = max(0, i - b)
        # 计算当前元素对应平滑窗口的结束下标
        end = min(size - 1, i + b)
        # 计算平滑窗口内元素的平均值，并赋值给平滑后的数组Z的对应位置
        Z[i] = sum(D[start:end + 1]) / (end - start + 1)
    return Z

def Identify_rising_edges(x, diff_frame, Tth):
    rawd = diff_frame[x]
    # 求rawd的转置
    rawd = rawd.T
   
    Spec_sm_pera = 5
    smd = smooth(rawd, Spec_sm_pera)
    d = rawd - smd

    # plt.figure(11)
    # plt.subplot(511)
    # plt.plot(x, rawd)
    # plt.subplot(512)
    # plt.plot(x, d)


    d, smd = RemoveHighPeaks(d, smd, Tth)
    # plt.subplot(513)
    # plt.plot(x, d)
    d, smd = RemoveNarrowPeaks(d, smd)
    # plt.subplot(514)
    # plt.plot(x, d)


    envd_1 = np.abs(hilbert(d))
    envd = smooth(envd_1, 17)

    temp_e = envd * 30
    temp = smooth(temp_e, 7)


    ID_behaviors = np.where(temp > Tth)[0]

    # plt.subplot(515)
    # plt.plot(x, temp)

  

    # plt.figure(2)
    # plt.subplot(4, 1, 1)
    # plt.plot(x, rawd, 'b')
    # plt.subplot(4, 1, 2)
    # plt.plot(x, d, 'b')
    # plt.plot(x, envd, 'r')
    # plt.ylim([-400, 400])
    # plt.xlabel('Time (s)')

    # plt.subplot(4, 1, 3)
    # plt.plot(x, temp, 'b')
    # plt.plot(x, np.ones_like(x) * Tth, 'r')
    # plt.ylim([0, 3 * Tth])
    # plt.xlabel('Time (s)')
    # plt.subplot(4, 1, 4)
    # plt.plot(x, smd, 'b')
    # plt.show()
    # plt.close()


    return smd, ID_behaviors


def SelectOutAllBehaviors(x, diff_frame, Tth):
    # 存储识别到的行为序列的列表
    ID_seg = []

    # 调用 Identify_rising_edges 函数识别上升边缘
    smd, ID_behaviors = Identify_rising_edges(x, diff_frame, Tth)

    # 如果识别到的行为序列不为空
    if len(ID_behaviors) > 0:
        # 计算相邻行为序列的差值
        diff_ID_behaviors = np.diff(ID_behaviors)
        

        # 找到差值大于1的索引位置
        K = np.where(diff_ID_behaviors > 1)[0]
        
        # 对K进行加1操作，得到新的索引位置
        K_N = K + 1

        # 构造起始索引数组
        A_f = np.concatenate(([1], K_N))

        # 构造结束索引数组
        A_s = np.concatenate((K, [len(ID_behaviors)]))

        # 将起始索引和结束索引组合成二维数组
        corres_index = np.column_stack((A_f, A_s))


        # 根据索引数组从ID_behaviors中提取对应的行为序列
        # corres_index的每个元素都减1，因为ID_behaviors的索引是从0开始的
        ID_seg = ID_behaviors[corres_index-1]


        # 获取ID_seg的维度
        _, ID_y = ID_seg.shape

        # 如果ID_seg只有一个维度，则进行转置操作
        if ID_y == 1:
            ID_seg = ID_seg.T

    return smd, ID_seg


def process_mkv(file_path):
    """
    定义处理单个 .mkv 文件的逻辑。
    :param file_path: str, .mkv 文件的路径
    """
    # 初始化局部变量参数
    
    fs = 30
    step = fs * 6
    Tth = 8000

    Res_all_all = []
    print(f"Processing file: {file_path}")
    # 计算帧间变化值
    diff_sums = compute_frame_differences(file_path)
    # 保存帧间变化值到csv文件，文件名以file_path的.mkv前面的部分作为文件名
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{filename}.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Difference'])
        for diff_sum in diff_sums:
            writer.writerow([diff_sum])
    # 根据运动信息检测函数提取指定的片段
    Nh = len(diff_sums) - 1
    Windows_num = int(Nh // step)
    for i in range(1, Windows_num + 1):
        x = list(range((i - 1) * step + 1, i * step + 1))
        smd, ID_seg = SelectOutAllBehaviors(x, diff_sums, Tth)
        # 将提取出的片段保存到Res_all_all列表中
        Res_all_all.append(ID_seg)
        # 根据ID_seg的前后片段裁剪视频，并保存到指定目录下
        for j in range(len(ID_seg)):
            start_time = int(fs * (x[ID_seg[j][0]] - 1))
            end_time = int(fs * (x[ID_seg[j][1]] - 1))
            subclip = VideoFileClip(file_path).subclip(start_time, end_time)
            output_file = f"./result/{filename}_{i}_{j+1}.mp4"
            subclip.write_videofile(output_file, codec='libx264', audio=False)
            print(f"Saved segment to {output_file}")
    return Res_all_all



if __name__ == "__main__":
    # 指定根目录
    root_directory = "d:/Tx2备份/test"  # 替换为实际路径
    # 打印根目录
    print("Root directory:", root_directory)

    find_and_process_mkv_files(root_directory)




