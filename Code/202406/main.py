import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import chenmin

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = chenmin.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showFullScreen()
    sys.exit(app.exec_())
