import subprocess
import psutil
from multiprocessing import Process
import time
import shlex



def record_process(directory):
    record_pid = []

    for i in range(4):
        print(i)
        """运行指定的sh脚本"""
        command = f'bash /home/ubuntu/chenminUI/video0_split.sh {i} {directory}'
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        x = psutil.Process(process.pid)
        while len(x.children()) == 0:
            time.sleep(0.1)  
        record_pid.append(x.children()[0].pid)
    #stop_recording(record_pid)
    #time.sleep(10000)

def stop_recording(pids):
    """停止所有录制进程"""
    for pid in pids:
        try:
            subprocess.Popen(["kill", "-9", str(pid)])
        except Exception as e:
            print(f"Error stopping process {pid}: {e}")
    print("All recording processes stopped.")

def run_inference(directory):
    print(directory)
    command_start = f'source /home/ubuntu/archiconda3/etc/profile.d/conda.sh && conda activate yolov5 && python ./inference-tool.py -i {directory} {directory[:-1]+"_output"}'
    pt = subprocess.Popen(command_start, shell=True, executable='/bin/bash')

if __name__ == "__main__":

    directory = "/mnt/Data/test"


    p = Process(target=record_process, args=(directory,))
#     ps = Process(target=run_inference, args=(directory,))
    p.start()
  #   time.sleep(10)
 #    ps.start()
    #print("Inference process started in a new process with PID: ", ps.pid)
    
    p.join()  

