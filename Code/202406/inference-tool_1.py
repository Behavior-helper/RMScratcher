"""
整个推理的入口，整合所有已知方法(除了sleap)
## 分步x    
1. 差分帧计算（可多进程）
2. 差分帧分类（需要分片）
3. pv分类（可多gpu多进程）
"""
import argparse
from datetime import datetime
from decord import VideoReader
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import time
import numpy as np
import pandas as pd
import multiprocessing
import functools
import re

import pickle
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from multiprocessing import Manager,Process
# from utils import clipSelected_once
# multiprocessing.set_start_method('forkserver', force=True)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

target_clas_num = 1
max_length = 60
displacement = 10
step = 60
windows_size = 60
area_min = 20
gray_thres = 60
inter = 10
intra = 10

def _make_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        The `argparse.ArgumentParser` that defines the CLI options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-r',
                        '--recursive',
                        action='store_true',
                        help='recursive find video')
    parser.add_argument("--video_suffix", type=str,default="mkv")
    parser.add_argument("-i", "--input_folder", type=str, help="input folder")
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        help=(
            "Path to trained model directory (with training_config.json). "
            "Multiple models can be specified, each preceded by --model."
        ),
    )

    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["none", "rich", "json"],
        default="rich",
        help=(
            "Verbosity of inference progress reporting. 'none' does not output "
            "anything during inference, 'rich' displays an updating progress bar, "
            "and 'json' outputs the progress as a JSON encoded response to the "
            "console."
        ),
    )
    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference only on CPU. If not specified, will use available GPU.",
    )
    device_group.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help=(
            "Run training on the i-th GPU on the system. If 'auto', run on the GPU with"
            " the highest percentage of available memory."
        ),
    )
    device_group.add_argument(
        "-p",
        "--process_num",
        type=int,
        default=4,
        help=(
            "Run training on the i-th GPU on the system. If 'auto', run on the GPU with"
            " the highest percentage of available memory."
        ),
    )
    parser.add_argument("output_folder", type=str,default="output_video")
    # Deprecated legacy args. These will still be parsed for backward compatibility but
    # are hidden from the CLI help.
    return parser

# 剪裁选中的视频片段，以帧为单位
def clipSelected_once(vid, dst, start, end):
    """
    从视频文件中截取指定帧范围并保存为新视频文件。

    Args:
        vid (cv2.VideoCapture): 视频文件读取对象。
        dst (str): 新视频文件的保存路径。
        start (int): 截取开始帧的索引（从1开始计数）。
        end (int): 截取结束帧的索引（包含该帧）。

    Returns:
        None

    Raises:
        AssertionError: 如果vid不是cv2.VideoCapture对象或end不大于start。

    """
    assert isinstance(vid, cv2.VideoCapture)
    assert end > start

    # 初始化
    # 设置指定帧
    vid.set(cv2.CAP_PROP_POS_FRAMES, max(start-1, 1))

    # 读取帧的数量
    num = end - start + 1

    # 获取视频帧的宽度和高度
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height), 1)

    # 遍历指定帧范围
    for i in range(num):
        succes, frame = vid.read()
        if not succes:
            break
        # 将读取的帧写入新的视频文件
        out.write(frame)

    # 释放视频写入对象
    out.release()


def cal_video(args_input):
    """
    计算视频帧的差分变化，并将结果写入csv文件。

    Args:
        args_input (tuple): 包含两个元素的元组，第一个元素为视频文件路径(str)，第二个元素为输出的csv文件路径(str)。

    Returns:
        None

    """
    src, dst_csv = args_input
    # 打印输入的视频文件路径
    # print("cal_video", src)
    # 如果输出的csv文件已存在，则直接返回，不进行计算
    # print("file exist, ignore compute it. ", src)
    if Path(dst_csv).exists():
        # print("file exist, ignore compute it. ", src)
        return
    try:
        # 打开视频文件
        vid = cv2.VideoCapture(src)
        # 读取视频的第一帧
        succes, prev = vid.read()
    except cv2.error as error:
        return

    if not succes:
        print('no succes')
        return
    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    # 打开csv文件，准备写入数据
    wrt = open(dst_csv, 'w+')
    count = 0
    launch = time.time()
    # 注释掉的部分代码，用于设置视频读取的起始位置（毫秒），这里未使用
    # vid.set(cv2.CAP_PROP_POS_MSEC, 60000)
    while vid.isOpened():
        # 读取视频的下一帧
        succes, frame = vid.read()
        if not succes:
            break
        # 将当前帧转换为灰度图像
        # 差分帧和颜色滤波合并
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 对原始图像作差
        # 对当前帧和前一帧的灰度图像进行差分
        # 对差分结果进行二值化处理
        #对原始图像作差
        diff = cv2.absdiff(gray, prev_gray)
        _, diff = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        # 定义腐蚀操作的卷积核
        kernel = np.ones((2, 2), np.uint8)
        # 对二值化后的差分图像进行腐蚀操作
        erode = cv2.erode(diff,kernel=kernel, iterations=2)
        # 查找腐蚀后的图像中的轮廓
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        erode_sum = 0
        erode_count = 0
        for idx, i in enumerate(contours):
            # 计算每个轮廓的面积
            area = cv2.contourArea(i)
            # 累加所有轮廓的面积
            erode_sum+= area
            # 累加轮廓的数量
            erode_count += 1

        # 计算差分图像中所有像素值的和
        diffsum = np.sum(diff)
        # 将计算得到的数据写入csv文件
        wrt.write("{}, {}, {}\n".format(erode_sum, diffsum, erode_count))
        # 更新前一帧的灰度图像为当前帧的灰度图像
        prev_gray = gray
        # 计数加一
        count = count + 1
    # 释放视频文件资源
    vid.release()

def filter_points(filename):
    df = pd.read_csv(filename, names=['erode_sum','diffsum', 'erode_count'],header=None)
    diff = df[(df['erode_sum'] > area_min * 2)]
    return [diff.index.values.tolist(), df]

def filter(points, min_length, intra,diff_df):
    res = []
    batch = {}
    info = {"avg": 0, "min": 1000000, "max":0, "count":1}
    for i in points:
        diff = diff_df.iloc[i,0]
        if 'end' in batch.keys() and i - batch['end'] < intra:
            batch['end'] = i
            info['avg'] = (info['avg'] + diff) / info['count']
            info['min'] = min(info['min'],diff)
            info['max'] = max(info['max'],diff)
        elif 'end' in batch.keys() and i - batch['end'] >= intra:
            if batch['end'] - batch['st'] >= min_length:
                info['avg'] = (info['avg'] + diff) / info['count']
                info['min'] = min(info['min'], diff)
                info['max'] = max(info['max'], diff)
                batch['info'] = info.copy()
                res.append(batch)
            batch = {}
            batch['st'] = i
            batch['end'] = i
            info = {"avg": 0, "min": 1000000, "max": 0, "count": 1}
        else:
            batch['st'] = i
            batch['end'] = i
            info = {"avg": diff, "min": diff, "max": diff, "count": 1}
        info['count'] += 1
    return res

def get_windows_dataframe(df,start,end,step,windows_size,max_length):
    # end-start是否少于windows_size
    if end - start <= windows_size - 1 and start + windows_size - 1 <= max_length:
        yield df.iloc[start: start + windows_size]
    else:
        for i in range(start,min(df.shape[0],end),step):
            ret_df = df.iloc[i:(i+windows_size),:]
            yield ret_df


def make_dataset_of_framediff(file_name, segment,csvdir,windows_size=60, step=60, seg_size=0,vid=None):
    csv_list = dict()
    location_list = []
    filename_stem = file_name.split('.')[0]
    signal_list = []
    signal_index_list = []
    all_time_id = []
    count = 0
    for row in segment:
        frame_60 = []
        start = row['st']
        end = row['end']
        csv_file = csvdir / file_name
        if csv_file not in csv_list:
            csv_list[csv_file] = pd.read_csv(csv_file,names=['erodesum','diffsum20','erode_sum'])
        csv = csv_list[csv_file]

        max_length = csv.shape[0]
        # 叠加
        for segment_60 in get_windows_dataframe(csv,start, min(end, max_length), step=step,windows_size=windows_size,max_length=max_length):
            start_index = segment_60.index.values[0]
            end_index = segment_60.index.values[-1]
            if end_index - start_index != windows_size - 1:
                continue
            dst = csvdir  / (filename_stem + '-' + str(start_index) + '-' + str(end_index) + '.mkv')
            # if not dst.exists():
            #     clipSelected_once(vid, str(dst), start_index, end_index)
            frame_60.append((start_index,end_index))
            signal_list.append(segment_60.to_numpy())
            signal_index_list.extend([count] * len(segment))
            all_time_id.extend([tid for tid in range(len(segment_60))])
            count += 1
        location_list.append((file_name.split('.')[0],frame_60))
    if len(signal_list) == 0:
        return 0, 0
    signal_np: np.ndarray = np.concatenate(signal_list, axis=0)
    signal_indexes = np.array(signal_index_list)
    signal_np = np.concatenate((signal_np,np.expand_dims(signal_indexes, axis=1)),axis=1)
    X = np.concatenate((signal_np,np.expand_dims(np.array(all_time_id),axis=1)),axis=1)
    X_df = pd.DataFrame(X, columns=['erodesum','diffsum20','erode_count','id','time'])
    X_df['id'] = X_df['id'].astype('int')
    return location_list, X_df

def extract_from_csv(csv_file: str):
    extraction_settings = ComprehensiveFCParameters()
    from utils import filter_points, filter
    pass_points, diff_df = filter_points(csv_file)
    test_list = []
    test_index_list = []
    segment = filter(pass_points, inter, intra, diff_df) 
    count = 0
    all_time_id = []
    info_segment = []
    for row in segment:
        seq: pd.core.series.Series = diff_df.iloc[row['st']:max(int(row['st'])+60,int(row['end'])),:]
        test_list.append(seq.to_numpy())
        test_index_list.extend([count] * len(seq))
        all_time_id.extend([tid for tid in range(len(seq))])
        count += 1
        info_segment.append(f"{row['st']}-{row['end']}")

    # 拼装pandas dataframe格式
    if len(test_list) == 0:
        return 0, 0
    test_np: np.ndarray = np.concatenate(test_list,axis=0)
    test_indexes = np.array(test_index_list)
    test_np = np.concatenate((test_np,np.expand_dims(test_indexes, axis=1)),axis=1)
    X_test = np.concatenate((test_np,np.expand_dims(np.array(all_time_id),axis=1)),axis=1)
    X_test = pd.DataFrame(X_test, columns=['erodesum','diffsum20','erode_count','diffsum10','id','time'])
    X_test['id'] = X_test['id'].astype('int')
    X_test = extract_features(X_test[['id','time','diffsum20','erodesum']], column_id='id', column_sort='time',
                        default_fc_parameters=extraction_settings,
                        # we impute = remove all NaN features automatically
                        impute_function=impute)
    
    return X_test, info_segment

def extract_feature_from_csv_and_segment(segment_list: list,csv_file: str,windows_size=60, step=60,src=None):
    """
    不切片
    Args:
    - segment, from filter_point
    - csv_file, str or path of cal_video output
    Outputs:
    - location_list
    """
    count = 0
    all_time_id = []
    location_list = [] #输出索引
    subset_list = [] # 用于暂时保存signal
    subset_index_list = [] #用于指定id
    subset_filename = [] # 用于定位视频片段
    csv_file = Path(csv_file)
    csv_df = pd.read_csv(csv_file, names=['erode_sum','diffsum', 'erode_count'],header=None)
    for row in segment_list:
        # 从文件名中寻找信息匹配
        filename_stem = csv_file.stem
        start = row['st']
        end = row['end']
        start = int(start)
        end = int(end)
        max_length = csv_df.shape[0]
        segment_str = f"{start}-{end}"
        for segment in get_windows_dataframe(csv_df,start, min(end, max_length), step=step,windows_size=windows_size,max_length=max_length):
            start_index = segment.index.values[0]
            end_index = segment.index.values[-1]
            if end_index - start_index != windows_size - 1:
                continue
            dst = csv_file.parent  / (filename_stem + '-' + str(start_index) + '-' + str(end_index) + '.mkv')
            # if not dst.exists():
            #     print("clip")
            #     clipSelected_once(vid, str(dst), start_index, end_index)
            location_tmp = {"start": start_index, "end": end_index, "filename": src,'id':count,'segment':segment_str, "csv_file":csv_file}
            location_list.append(location_tmp)
            subset_list.append(segment.to_numpy())
            subset_index_list.extend([count] * len(segment))
            all_time_id.extend([tid for tid in range(len(segment))])
            count += 1
    if len(subset_list) == 0:
        return 0, 0
    subset_np: np.ndarray = np.concatenate(subset_list,axis=0)
    subset_indexes = np.array(subset_index_list)
    X = np.concatenate((subset_np,np.expand_dims(subset_indexes, axis=1)),axis=1)
    # 拼装pandas dataframe格式
    X = np.concatenate((X,np.expand_dims(np.array(all_time_id),axis=1)),axis=1)
    X_df = pd.DataFrame(X, columns=['erodesum','diffsum20','erode_count','id','time'])
    X_df['id'] = X_df['id'].astype('int')
    location_pd = pd.DataFrame(location_list)
    return X_df, location_pd

def predict_signal(X_df, location_pd,clf):
    extraction_settings = ComprehensiveFCParameters()
    # # print(location_pd)
    # # print(X_df)
    feature = extract_features(X_df[['id','time','diffsum20']], column_id='id', column_sort='time',
                        default_fc_parameters=extraction_settings,
                        impute_function=impute,n_jobs=0)
    label_pr = clf.predict(feature)
    location_pd['predict'] = label_pr
    # location_pd['predict'] = 0
    return location_pd

def step2(args_input):
    """一个单独的实例,提取出差分帧信号并分类,针对单个cal_video csv文件"""
    src, dst_csv = args_input
    dst_csv = Path(dst_csv)
    vid = cv2.VideoCapture(src)
    pass_points, diff_df = filter_points(dst_csv)
    segment = filter(pass_points, inter, intra, diff_df) 
    X_df, location = extract_feature_from_csv_and_segment(segment, dst_csv, windows_size=60, step=60,src=src)
    # print(X_df, location)
    vid.release()
    # 信号分类算法
    return [X_df, location]

def step3(location,clas,args_input,video_filename):
    print("step")
    if location is None:
        print("step location")

        return
    src, dst_csv = args_input
    dst_csv = Path(dst_csv)
    if (dst_csv.parent / (dst_csv.stem + "_res.csv")).exists():
        return
    vid = cv2.VideoCapture(str(video_filename))
    tmp_video_list = []
    for i in location.index:
        # print(i)
        filename = Path(location.loc[i,'filename'])
        
        start = location.loc[i,'start']
        end = location.loc[i,'end']
        count = 0
        judge_list = []
        judge_score_list = []

        dst = dst_csv.parent  / (filename.stem + '-' + str(start) + '-' + str(end) + '.mkv')
        # print(filename, dst)
        if not dst.exists():
            clipSelected_once(vid,str(dst),start,end)
        tmp_video_list.append(str(dst))
    results = clas.predict_batch(tmp_video_list)
    for result,i in zip(results,location.index):
        idmax =  int(result['class_ids'][0])
        score =  result['scores'][0]
        location.loc[i,'score'] = score
        location.loc[i, 'behavor'] = idmax # 0 for normal, 1 for scratching
        if idmax == target_clas_num:  # 如果是想要的抓挠类
            count += 1
    location.to_csv(dst_csv.parent / (dst_csv.stem + "_res.csv"),mode = 'w')
    vid.release()

def clip_predict(args_input):
    src, dst_csv = args_input
    dst_csv = Path(dst_csv)

def cpu_process(queue, i):
    """
    cpu工作，包含calvideo和后面的"""
    # print("begin cpu_process", queue, i)
    with open('20221201-v4-ds20221118.pickle', 'rb') as fw:
        clf = pickle.load(fw)
    cal_video(i)
    src, dst_csv = i
    dst_csv = Path(dst_csv)
    src = Path(src)
    pass_points, diff_df = filter_points(dst_csv)
    segment = filter(pass_points, inter, intra, diff_df) 
    video_filename = ''
    if (dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv")).exists():
        location_pd = pd.read_csv(dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv"))
        queue.put([i, video_filename])
    else:
        X_df, location = extract_feature_from_csv_and_segment(segment, dst_csv, windows_size=60, step=60, src=src)
        if isinstance(X_df, int):
            print(f"X_df {src} \n")
            queue.put([i, video_filename])
            return
        location_pd = predict_signal(X_df, location,clf)
        if location_pd.shape[0] == 0:
            print("location \n")
            queue.put([i, video_filename])
            return
        video_filename = Path(location_pd.loc[0,'filename'])
        if (dst_csv.parent / (dst_csv.stem + "_location_sig_sc.csv")).exists():
            queue.put([i, video_filename])
            return
        vid = cv2.VideoCapture(str(video_filename))
        for j in location.index:
            filename = Path(location.loc[j,'filename'])            
            start = location.loc[j,'start']
            end = location.loc[j,'end']
            dst = dst_csv.parent  / (filename.stem + '-' + str(start) + '-' + str(end) + '.mkv')
            # print(filename, dst)
            if not dst.exists():
                clipSelected_once(vid,str(dst),start,end)
        location_pd.to_csv(dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv"))
        queue.put([i, video_filename])
    # print('cpu part log: ', i ,video_filename, queue.empty())

class Func(object):
    def __init__(self):
        # 利用匿名函数模拟一个不可序列化象
        # 更常见的错误写法是，在这里初始化一个数据库的长链接
        self.num = lambda: None

    def work(self, num=None):
        self.num = num
        return self.num

    @staticmethod
    def call_back(res):
        print(f'Hello,World! {res}')

    @staticmethod
    def err_call_back(err):
        print(f'出错啦~ error：{str(err)}')

def generate_result(root_path):
    """
    按照名字生成表格
    """
    total_df = []
    root_path = Path(root_path)
    for i in root_path.rglob("*_res.csv"):
        target_ret = pd.read_csv(i)
        if isinstance(target_ret,int):
            continue
        total_df.append(target_ret)
    total_df = pd.concat(total_df,ignore_index=True)
    total_df = total_df[total_df['behavor']== 1] 
    total_df = total_df.groupby('filename', as_index=False).apply(lambda df:df.drop_duplicates("segment"))
    total_df = total_df.reset_index()
    def extract_group(row):
        stem = Path(row).stem
        filename_stem, camera_part, segment = re.search("(\d+)-(\d)_(\d+)", stem).groups()
        return filename_stem

    def extract_camera(row):
        stem = Path(row).stem
        filename_stem, camera_part, segment = re.search("(\d+)-(\d)_(\d+)", stem).groups()
        return camera_part
    def extract_part(row):
        stem = Path(row).stem
        filename_stem, camera_part, segment = re.search("(\d+)-(\d)_(\d+)", stem).groups()
        return segment
    total_df['group'] = total_df['filename'].apply(extract_group)
    total_df['camera'] = total_df['filename'].apply(extract_camera)
    total_df['part'] = total_df['filename'].apply(extract_part)
    total_excel = []
    for idx, i in enumerate(list(total_df['group'].unique())):
        print(i)
        now = total_df[total_df['group']==i]
        # print(now[['filename','part','camera']])
        table = now.pivot_table(index='camera',columns='part',values='filename',aggfunc='count',fill_value=0,margins = True, margins_name='Total')
        print(table)
        table['date'] = i
        total_excel.append(table['Total'])
    # print(pd.DataFrame(total_excel))


def main_split(args: list = None):
    """Entrypoint for `MOUSEACTION` CLI for running inference.

    Args:
        args: A list of arguments to be passed into mouse-action.
    """
    start_timestamp = str(datetime.now())
    logger.info("Started inference at:", start_timestamp)
    parser = _make_cli_parser()
    args, _ = parser.parse_known_args(args=args)
    logger.info("Args:")
    logger.info(vars(args))
    input_path = Path(args.input_folder)
    if input_path.is_file():
        video_list = [input_path]
    else:
        video_list = input_path.rglob(f"*.{args.video_suffix}") \
            if args.recursive \
            else input_path.glob(f"*.{args.video_suffix}")
    # 组装多进程的任务入参
    list_args = []
    video_list = [i for i in video_list]
    for video_filename in video_list:
        workdir =  video_filename.parent
        workdir = Path(str(workdir).replace(str(input_path),args.output_folder))
        workdir.mkdir(exist_ok=True,parents=True)
        newcsv_file = workdir / (video_filename.stem + ".csv")
        list_args.append([str(video_filename),str(newcsv_file)])
    
    queue = Manager().Queue()

    cpu_pool = multiprocessing.Pool(processes=4)
    for i in list_args:
        cpu_pool.apply_async(cpu_process,args=(queue,i),error_callback=Func.err_call_back)
    from ppvideo import PaddleVideo
    model_str = "ppTSM20221208/ppTSM_mouse_20220613"
    clas = PaddleVideo(model_file= model_str +'.pdmodel',
                    params_file= model_str  + '.pdiparams',
                    use_gpu=True,use_tensorrt=False,batch_size=4)
    count = 0
    dest_count = len(list_args)
    while True:
        if count == dest_count:
            break
        if not queue.empty():
            # print("init inference")
            arg_item, video_filename = queue.get(True)
            dst_csv = Path(arg_item[1])
            src = Path(arg_item[0])
            count += 1
            print(f"dest is {dest_count } count is {count} {dst_csv} {video_filename}")
            if src == '':
                continue
            if (dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv")).exists():
                location = pd.read_csv(dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv"))
                # print(arg_item, video_filename, location.loc[0,'filename'])
                scratching_location = location[location['predict'] == target_clas_num]
                step3(scratching_location, clas,arg_item, src)
            else:
                print("video _location_sc don't exist", dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv"))
                # step3(scratching_location, clas,arg_item, video_filename)
        else:
            time.sleep(5)
    cpu_pool.close()
    cpu_pool.join()
    generate_result(args.output_folder)
    # for i in range(1,8):
    #     print(f"day{i}")
    #     generate_result(f"G:\\NanShan\\SRC-res-trial3\\LXA4\\LXA4_DAY{i}")

if __name__ == "__main__":
    main_split()
    # main()
    print("\n")