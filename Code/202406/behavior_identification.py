import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.signal import hilbert
from extraxtSignalFeaturesFunc import medfreq, peak2rms, clearance_factor, crest_factor
from extraxtSignalFeaturesFunc import impulse_factor
from Yolov8_classfication import Yolov8_classfication
from Scratching_regions import behavior_predict
import csv
import time
import cv2

def findmdPeaks(md):

    md = smooth(md, 15)


    if len(md) < 3:
        return [], []


    peak_idx, _ = find_peaks(md)
    peak_value = md[peak_idx]


    high_peak_idx = peak_idx[peak_value > 2e3]
    high_peak_value = peak_value[peak_value > 2e3]
    
    if high_peak_value.size == 0:
        return [], []

    pPks = np.max(high_peak_value)
    pID = np.argmax(high_peak_value)
    pid = high_peak_idx[pID]


    if len(high_peak_value) > 1:
        idPID = np.where(np.diff(high_peak_idx) > 10)[0]
        if idPID.size > 0:
            idP1 = np.concatenate(([high_peak_idx[0]], high_peak_idx[idPID + 1]))
            idP2 = np.concatenate((high_peak_idx[idPID], [high_peak_idx[-1]]))
            pPks = []
            pid = []
            for fhigh_peak_idx in range(len(idP1)):
                local_max = np.max(md[int(idP1[fhigh_peak_idx]):int(idP2[fhigh_peak_idx])+1])
                local_max_idx = np.argmax(md[int(idP1[fhigh_peak_idx]):int(idP2[fhigh_peak_idx])+1])
                pPks.append(local_max)
                pid.append(int(idP1[fhigh_peak_idx]) + local_max_idx)
    return pPks, pid



def RemoveNarrowPeaks(d, md):
    pPks, pid = findmdPeaks(md)
    

    if not isinstance(pPks, (list, np.ndarray)):

        pPks = [pPks]
    

    if not isinstance(pid, (list, np.ndarray)):

        pid = [pid]

    #print(len(pPks))

    if len(pPks) == 0 or len(pid) == 0:
        return d, md
    if len(pPks) == 1 or len(pid) == 1:
        return d, md
    
    Npks = len(pPks)

    for i in range(Npks):

        idhpk1 = max(0, pid[i] - 20)
        idhpk2 = min(len(d), pid[i] + 21)
        idhf = np.where(md[idhpk1:idhpk2] > pPks[i]/3)[0]
        
        if len(idhf) < 20 and Npks > 1:  
            d[idhpk1:idhpk2] = 10
            md[idhpk1:idhpk2] = 10

    return d, md



def RemoveHighPeaks(d, md, Tth):
    pPks, pid = findmdPeaks(md)

    if not isinstance(pPks, (list, np.ndarray)):

        pPks = [pPks]
    
    if not isinstance(pid, (list, np.ndarray)):

        pid = [pid]


    if len(pPks) == 0 or len(pid) == 0:
        return d, md
    
    if len(pPks) == 1 or len(pid) == 1:
        return d, md
    
    hpThres = Tth * 2
    pPks = np.array(pPks)
    idH = np.where(pPks > hpThres)[0]

    for itemp in range(len(idH)):
        idHtemp = idH[itemp]
        idhpk1 = max(0, pid[idHtemp] - 20)
        idhpk2 = min(len(d), pid[idHtemp] + 21)
        d[idhpk1:idhpk2] = 10
        md[idhpk1:idhpk2] = 10

    return d, md

def smooth(D, k):

    size = len(D)

    Z = np.zeros(size)

    b = (k - 1) // 2

    for i in range(size):

        start = max(0, i - b)

        end = min(size - 1, i + b)

        Z[i] = sum(D[start:end + 1]) / (end - start + 1)
    return Z

def Identify_rising_edges(x, diff_frame, Tth):
    rawd = diff_frame[x]

    rawd = rawd.T
   
    Spec_sm_pera = 5
    smd = smooth(rawd, Spec_sm_pera)
    d = rawd - smd



    d, smd = RemoveHighPeaks(d, smd, Tth)

    d, smd = RemoveNarrowPeaks(d, smd)



    envd_1 = np.abs(hilbert(d))
    envd = smooth(envd_1, 17)

    temp_e = envd * 30
    temp = smooth(temp_e, 7)


    ID_behaviors = np.where(temp > Tth)[0]



    return smd, ID_behaviors


def SelectOutAllBehaviors(x, diff_frame, Tth):

    ID_seg = []


    smd, ID_behaviors = Identify_rising_edges(x, diff_frame, Tth)


    if len(ID_behaviors) > 0:

        diff_ID_behaviors = np.diff(ID_behaviors)
        


        K = np.where(diff_ID_behaviors > 1)[0]
        

        K_N = K + 1


        A_f = np.concatenate(([1], K_N))


        A_s = np.concatenate((K, [len(ID_behaviors)]))


        corres_index = np.column_stack((A_f, A_s))



        ID_seg = ID_behaviors[corres_index-1]



        _, ID_y = ID_seg.shape


        if ID_y == 1:
            ID_seg = ID_seg.T

    return smd, ID_seg

def Identify_behaviors(csv_path,Types_of_behaviors, ID_seg, smd, x, Identify_scraching_region,cap):
    if Types_of_behaviors[0]:
        Res_all = Identify_scrach(csv_path,ID_seg, smd, x, Identify_scraching_region,cap)
        #print(Res_all)
    return Res_all

def Identify_scrach(csv_path,ID_seg, smd, x, Identify_scraching_region, cap):
    """
    Identify_scrach is used to identify scratching behaviors.

    Args:
        ID_seg (numpy.ndarray): Segmentation indices.
        smd (numpy.ndarray): Differential signal.
        x (numpy.ndarray): Time vector.
        Name_data_vid (str): Name of the video data.
        Identify_scraching_region (bool): Flag to identify scratching region.

    Returns:
        tuple: A tuple containing:
            Num_scrach (int): Number of scratching behaviors.
            ID_scrach (numpy.ndarray): Indices of scratching behaviors.
            ID_segments_scrach (numpy.ndarray): Segments corresponding to scratching behaviors.
            Res_all (list): Scratch detection results.
    """
    # Initialize default parameters
    Num_scrach = 0
    ID_scrach = []
    ID_segments_scrach = []
    Res_all = []

    ID_seg_array = np.array(ID_seg)

    if len(ID_seg_array.shape) == 1:

        ID_seg_M = len(ID_seg_array)

    else:

        ID_seg_M, _ = ID_seg_array.shape


    # Run the differential signal for each row
    for i in range(ID_seg_M):
        Res_all_part = []
        ID_seg_row = ID_seg[i, :]
        part_smd = smd[ID_seg_row[0]:ID_seg_row[1] + 1]
        part_x = x[ID_seg_row[0]:ID_seg_row[1] + 1]

        # Visualize the selected fragments (optional)
        # plt.figure(10 + i)
        # plt.plot(part_x, part_smd, 'g')
        # plt.pause(0.1)

        # Check if part_smd is empty
        if len(part_smd) >= 3:
            # print(ID_seg)
            # print(ID_seg_row)
            # print(ID_seg_M)
            # print(part_smd)
            # print(part_x)
            Res_all_part = two_stage_algorithm(csv_path,part_smd, part_x, Identify_scraching_region, cap)
            #print(Res_all_temp)
        else:
            Res_all_part = []
        

        if len(Res_all_part) != 0 and Res_all_part[2] >= 1:
            Res_all.append(Res_all_part) 
    
    #print(Res_all)
    return Res_all


def identify_touch_duration(sec_id, part_smd, part_x):

    num_touch = 0
    locs = pks = w = p = np.array([])

    duration_bout = (sec_id[1]- sec_id[0]) / 120


    if (sec_id[1]- sec_id[0]) >= 3: 

        #print(part_smd)
        locs_p, _ = find_peaks(part_smd)
        pks = part_smd[locs_p]
        
        part_x = np.array(part_x)
        locs = np.array(locs)
        locs = part_x[locs_p.astype(int)]
 
        w, p, _, _ = peak_widths(part_smd, locs_p)
        # print(locs)
        # print(w)
        # print(p)
   
    pks_feature = np.vstack((pks, locs, p, w)).T

    

    for feature in pks_feature:
        if abs(feature[2]) > 40 and feature[3] > 2:  
            num_touch += 1

    return sec_id, locs, num_touch, duration_bout

def two_stage_algorithm(csv_path,part_smd, part_x, Identify_scratching_region, cap):
    # Initialize variables
    Res_all_temp = []  # Results after filtering
    Res_potential = []  # Potential signal segments
    Res_exact = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
    locs = []
    scratch_region = 404
    pks_high = 150000  # Highest peak threshold
    pks_low = 40  # Lowest peak threshold
    num_valid_peaks_thres = 3  # Number of usable peaks for scratching behavior
    Everage_peak_interval_thres = 20  # Average interval between peaks during scratching


    # Find peak features in the time domain
    locs, _ = find_peaks(part_smd)  # Find peak positions
    pks = part_smd[locs]
    num_valid_peaks = np.where((pks > pks_low) & (pks < pks_high))[0]  # Find usable peaks
    peak_interval = np.diff(locs)  # Calculate peak intervals
    Everage_peak_interval = np.mean(peak_interval)  # Calculate average peak interval
    
    # Processed signal segment ID
    Sec_ID = [part_x[0], part_x[-1]]  # ID of segments that meet the threshold
    
    # Extract signal features
    freq_fs = medfreq(part_smd, 360)
    Diff_peaks = peak2rms(part_smd)
    sig_rms = np.sqrt(np.mean(part_smd**2))
    sig_shape_factor = sig_rms / np.abs(np.mean(part_smd))
    crest_factor_value = crest_factor(part_smd)

    impulse_factor_value = impulse_factor(part_smd)

    clearance_factor_value = clearance_factor(part_smd)

    if (len(num_valid_peaks) >= num_valid_peaks_thres) and (Everage_peak_interval < Everage_peak_interval_thres):

        Sec_ID_paras = Sec_ID + [freq_fs, Diff_peaks, sig_rms, sig_shape_factor, impulse_factor_value,  clearance_factor_value, crest_factor_value]

        if (freq_fs < 50) and (Diff_peaks < 5) and (2000 >sig_rms > 600) and (sig_shape_factor < 2) and (impulse_factor_value < 5) and (clearance_factor_value < 18) and (crest_factor_value  < 5):
            mkv_file_path = csv_path.replace(".csv", ".mkv")

            if (freq_fs < 30) and (Diff_peaks < 3) and (1500>sig_rms > 600) and (sig_shape_factor < 1.5) and (impulse_factor_value < 3.5) and (clearance_factor_value < 6) and (crest_factor_value  < 3.5):

                Sec_ID, locs, Num_touch, duration_bout = identify_touch_duration(Sec_ID, -part_smd, part_x)
                if Identify_scratching_region == 1 and Num_touch >= 1:
                    scratch_region = behavior_predict(mkv_file_path, locs)
                Res_exact = Sec_ID + [Num_touch, duration_bout, scratch_region]
            else:
                Sec_ID, locs, Num_touch, duration_bout = identify_touch_duration(Sec_ID, -part_smd, part_x)

            ID_seg = list(range(Sec_ID[0], Sec_ID[1]))
            scratch_behavior = 0
            if Num_touch >= 1:

                scratch_behavior = Yolov8_classfication(mkv_file_path,ID_seg,cap)

            if scratch_behavior == 1:
                if Identify_scratching_region == 1:
                    scratch_region = behavior_predict(mkv_file_path, ID_seg)
                Res_exact = Sec_ID + [Num_touch, duration_bout, scratch_region]
                    
            Res_exact = np.array(Res_exact)

    if np.isnan(Res_exact).all() == 0 :
        Res_all_temp = Res_exact 
    return Res_all_temp


def Behaviors_identification_by_videos(csv_path,Identify_scraching_region):


    print("Processing: " + csv_path)

    data = pd.read_csv(csv_path,engine = "python")

    diff_frame = data.iloc[:, 0]

    Nh = len(diff_frame) - 1
    fs = 30
    step = fs * 6
    Tth = 8000
    Windows_num = Nh // step
    time_5min = []
    para_all = []
    Option_identify_scrach = True
    Option_identify_shake = False
    Types_of_behaviors = [Option_identify_scrach, Option_identify_shake]
    mkv_file_path = csv_path.replace(".csv", ".mkv")
    cap = cv2.VideoCapture(mkv_file_path)

    for i in range(1, Windows_num + 1):
        Res_all_all = []
        x = list(range((i - 1) * step + 1, i * step + 1))
        smd, ID_seg = SelectOutAllBehaviors(x, diff_frame, Tth)
        Res_all_all = Identify_behaviors(csv_path,Types_of_behaviors, ID_seg, smd, x, Identify_scraching_region,cap)
        if len(Res_all_all) != 0 and Res_all_all != []:
            time_5min = time_5min + Res_all_all
    time_5min = np.array(time_5min)

    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        rows = list(csv_reader)
        fourth_column = [row[1] for row in rows]


    for i in range(time_5min.shape[0]):
        start_frame = int(time_5min[i, 0])
        end_frame = int(time_5min[i, 1])
        if start_frame < len(fourth_column) and end_frame < len(fourth_column):
            time_5min[i, 0] = fourth_column[start_frame]
            time_5min[i, 1] = fourth_column[end_frame]
    return time_5min


if __name__ == "__main__":

    csv_path = "/home/ubuntu/chenminUI/output_folder/202305111822-0_01.csv"

    Identify_scraching_region = False
    start_time = time.time()
    time_5min = Behaviors_identification_by_videos(csv_path,Identify_scraching_region)
    end_time = time.time()

    print("Time cost: " + str(end_time - start_time) + "s")
    

    print(time_5min)
    print(time_5min.shape)