import os
import cv2
import numpy as np
import csv
import time
import tempfile
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda import gpuarray
import cProfile

# CUDA Kernel for grayscale conversion and frame difference calculation
kernel_code = """
__global__ void convert_and_process_frames(unsigned char *current_frame, unsigned char *prev_frame, unsigned int *result, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Convert to grayscale using weighted sum
        unsigned char current_gray = 0.2989f * current_frame[3*idx] + 0.5870f * current_frame[3*idx+1] + 0.1140f * current_frame[3*idx+2];
        unsigned char prev_gray = prev_frame[idx];

        // Calculate difference and update result
        unsigned char diff = abs(current_gray - prev_gray);
        if (diff > 20) {
            atomicAdd(result, 1);
        }

        // Store the grayscale value for the next iteration
        prev_frame[idx] = current_gray;
    }
}
"""

def create_video_capture_with_gst_pipeline(video_path):
    pipeline = (
        f"filesrc location={video_path} ! "
        "qtdemux ! "
        "h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "appsink sync=false drop=true max-buffers=240"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

def calculate_frame_differences(input_video, output_csv):
    try:
        vid = create_video_capture_with_gst_pipeline(input_video)
        fps = vid.get(cv2.CAP_PROP_FPS)
        success, prev = vid.read()
        if not success:
            print("No success in reading video frames.")
            return
    except cv2.error as error:
        print("Error reading video:", error)
        return

    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = frame_width * frame_height

    # Define block and grid dimensions
    block_dim = (16, 16, 1)
    grid_dim = (frame_width // block_dim[0] + 1, frame_height // block_dim[1] + 1, 1)

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    temp_file_name = temp_file.name
    csv_writer = csv.writer(temp_file)

    # Compile the CUDA kernel
    mod = compiler.SourceModule(kernel_code)
    convert_and_process_frames = mod.get_function("convert_and_process_frames")

    # Pre-allocate GPU memory
    d_prev_gray = gpuarray.empty((frame_size,), dtype=np.uint8)
    d_current_frame = gpuarray.empty((frame_size * 3,), dtype=np.uint8)
    d_result = gpuarray.zeros((1,), dtype=np.uint32)

    # Allocate pinned memory for frame data on CPU
    pinned_frame = cuda.pagelocked_zeros((frame_size * 3,), dtype=np.uint8)
    
    # Allocate a CPU buffer for result
    result_buffer = cuda.pagelocked_zeros((1,), dtype=np.uint32)

    # Initialize CUDA stream
    stream = cuda.Stream()

    count = 0
    start_time_diff = time.time()

    # Convert the first frame to grayscale on CPU, then transfer it to GPU
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY).flatten()
    d_prev_gray.set(prev_gray)

    while True:
        success, frame = vid.read()
        if not success:
            break

        # Copy frame to pinned memory and flatten it
        np.copyto(pinned_frame, frame.flatten())

        # Asynchronously transfer the current frame to GPU
        cuda.memcpy_htod_async(d_current_frame.gpudata, pinned_frame, stream)

        # Reset result for the current frame difference calculation
        d_result.fill(0)

        # Asynchronously execute the kernel
        convert_and_process_frames(d_current_frame, d_prev_gray, d_result, np.int32(frame_width), np.int32(frame_height),
                                   block=block_dim, grid=grid_dim, stream=stream)

        # Asynchronously copy the result back to CPU
        cuda.memcpy_dtoh_async(result_buffer, d_result.gpudata, stream)

        # Synchronize the stream to ensure kernel execution is complete
        stream.synchronize()

        # Get the result
        result = result_buffer[0]

        # Get the timestamp and write to CSV
        timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        csv_writer.writerow([result, timestamp])

        count += 1

    vid.release()
    temp_file.close()

    end_time_diff = time.time()
    run_time_diff = end_time_diff - start_time_diff
    print(f"Processed {count} frames. CSV file saved as {output_csv}")

    os.rename(temp_file_name, output_csv)

if __name__ == "__main__":
    cProfile.run('calculate_frame_differences("/home/ubuntu/chenminUI/testvideo/202305111822-0_01.mkv", "/home/ubuntu/chenminUI/testvideo/202305111822-0_01.csv")', 'profile_output.prof')



