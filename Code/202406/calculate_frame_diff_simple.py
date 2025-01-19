
import os
import cv2
import numpy as np
import csv
import time
import tempfile
import cProfile

def calculate_frame_differences(input_video, output_csv):
    try:
        gst_pipeline = (
            f"filesrc location={input_video} ! "
            "qtdemux ! h264parse ! nvv4l2decoder ! nvvideoconvert ! "
            "video/x-raw, format=(string)GRAY8 ! appsink sync=false drop=true max-buffers=100"
        )
        vid = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)   
        success, prev_gray = vid.read()
        if not success:
            print("No success in reading video frames.")
            return
    except cv2.error as error:
        print("Error reading video:", error)
        return

    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    temp_file_name = temp_file.name
    csv_writer = csv.writer(temp_file)

    count = 0
    start_time_diff = time.time()
    while vid.isOpened():
        success, gray = vid.read()
        if not success:
            break

        timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        diff = cv2.absdiff(gray, prev_gray)
        _, diff = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        prev_gray = gray
        diff_sum = np.sum(diff)
        csv_writer.writerow([diff_sum, timestamp])
        count += 1

    vid.release()
    temp_file.close()

    end_time_diff = time.time()
    run_time_diff = end_time_diff - start_time_diff
    print(f"Processed {count} frames. CSV file saved as {output_csv}")

    os.rename(temp_file_name, output_csv)

if __name__ == "__main__":
    cProfile.run('calculate_frame_differences("/home/ubuntu/chenminUI/testvideo/202305111822-0_01.mkv", "/home/ubuntu/chenminUI/testvideo/202305111822-0_01.csv")', 'profile_output.prof')
