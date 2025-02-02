import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
 
class TimmerDemo(QWidget):
    def __init__(self, parent=None):
        super(TimmerDemo, self).__init__(parent)
        self.setWindowTitle("TimmerDemo")
        self.time = 5
 
        self.label = QLabel("<font color=red size=128 > <b>(%s)s</b>" % (self.time))
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint) 
 

        self.timer = QTimer()
        self.timer.timeout.connect(self.showtime)
 
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 1, 2)
 
        self.setLayout(layout)
        self.starttimer()
 
    def showtime(self):
        self.time -= 1
        self.label.setText("<font color=red size=128 > <b>(%s)s</b>" % (self.time))
        if self.time == 0:
            self.endtimer()
        
 
    def starttimer(self):
        self.timer.start(1000)
 
 
    def endtimer(self):
        self.timer.stop()
        self.close()
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    listwidget = TimmerDemo()
    listwidget.show()
    sys.exit(app.exec_())