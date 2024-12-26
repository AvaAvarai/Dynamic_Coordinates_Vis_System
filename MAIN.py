"""
DCVis - Dynamic Coordinate Visualization System

Requirements:
- Python 3.x: Recommended Python 3.6 or higher for compatibility.
- Libraries installed from requirements.txt: Use pip to install the required libraries. Run the following command in the terminal: pip install -r requirements.txt

Usage:
Execute this script to launch the DCVis application: python DCVIS_MAIN.py
"""

from PyQt6 import QtWidgets
import sys
import VIEW, CONTROLLER

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    view = VIEW.View()
    controller = CONTROLLER.Controller(view)
    view.controller = controller

    view.show()
    app.exec()
