import os
import cv2
import numpy as np
import csv
import time
import tempfile

def calculate_frame_differences(input_video, output_csv):
    try:
        vid = cv2.VideoCapture(input_video)
        fps = vid.get(cv2.CAP_PROP_FPS)
        success, prev = vid.read()
        if not success:
            print("No success in reading video frames.")
            return
    except cv2.error as error:
        print("Error reading video:", error)
        return

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    temp_file_name = temp_file.name
    csv_writer = csv.writer(temp_file)

    count = 0
    start_time_diff = time.time()
    while vid.isOpened():
        success, frame = vid.read()
        if not success:
            break

        timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, diff = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        erode = cv2.erode(diff, kernel=kernel, iterations=2)
        contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        erode_sum = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 20)
        erode_count = sum(1 for contour in contours if cv2.contourArea(contour) > 20)

        diff_sum = np.sum(diff)
        prev_gray = gray

        csv_writer.writerow([erode_sum, diff_sum, erode_count, timestamp])
        count += 1

    vid.release()
    temp_file.close()

    end_time_diff = time.time()
    run_time_diff = end_time_diff - start_time_diff
    print(f"Processed {count} frames. CSV file saved as {output_csv}")

    os.rename(temp_file_name, output_csv)

if __name__ == "__main__":
    import glob

    def process_all_mkv_files_in_folder(folder_path, output_folder):
        mkv_files = glob.glob(os.path.join(folder_path, "*.mkv"))
        
        for mkv_file in mkv_files:
            base_name = os.path.basename(mkv_file)
            output_csv = os.path.join(output_folder, base_name.replace(".mkv", ".csv"))
            calculate_frame_differences(mkv_file, output_csv)

    folder_path = "/home/ubuntu/chenminUI/testvideo"
    output_folder = "/home/ubuntu/chenminUI/testvideo"
    process_all_mkv_files_in_folder(folder_path, output_folder)
