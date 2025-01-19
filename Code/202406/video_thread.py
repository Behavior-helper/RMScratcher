import threading
import os
class video_handler(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.mEvent = threading.Event()
    def pause(self):
        print("pause")
        self.mEvent.clear()
    def resume(self):
        print("resume")
        self.mEvent.set()
    def run(self):
        print("runit!!!!")
        os.system("./three.sh")