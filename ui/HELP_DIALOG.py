from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Help")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        header = QLabel("Keyboard Controls and Shortcuts")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        controls_info = """
          • UI Controls:
            F1: Load CSV/TXT Dataset from file
            F2: Recenter visualization plot
            F3: Create visualization plot
            F4: Refresh visualization plot
            ESC: Exit DCVis application
            
          • Selected Data Points:
            Q: Roll (select next) selected data points backward
            E: Roll (select previous)selected data points forward
            W: Move selected data points up
            S: Move selected data points down
            D: Delete selected data points
            C: Clone selected data points
            I: Insert new data point of chosen class
            P: Print the selected data points details to the console
            
          • Synthetic Data Generation:  
            G: Generate a specified number of data points with CTGAN over chosen epochs
            ?: Infers the class of the highlighted case(s) with a vote of standard machine learning models
            R: Relabel the selected data points with a chosen class
              
          • Visualization Plot Interaction:
            Left Click: Select and highlight data points.
            Right Click: Set clipping boundaries or clear data.
            Middle Click and Drag: Pan the plot.
            Middle Click and Hold: Grow selection box.
            Scroll Wheel: Zoom in and out on the plot.
        
        For deleting associative rules can right click and delete individual rules or click the clear all rules button.
        
        Associative rules interface and classification interface is still a work in progress.
        
        All other controls are available through the GUI.
        Please see the README.md file in the DCVis project repsitory for more information.
        
        If you encounter any problems please let us known through our Github issues page, thank you!
        """
        
        label = QLabel(controls_info)
        label.setStyleSheet("font-size: 14px;")
        layout.addWidget(label)
        button = QPushButton("Close", clicked=self.accept)
        layout.addWidget(button)
        
        self.setLayout(layout)
