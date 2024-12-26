from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Help")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        controls_info = """
        Keyboard Shortcuts for UI Controls:
          F1: Load Dataset
          F2: Recenter Plot
          F3: Create Plot
          F4: Refresh Plot
          ESC: Exit Application
          
        Keyboard Controls for Data Points:
          Q: Roll selected data points backward
          E: Roll selected data points forward
          W: Move selected data points up
          S: Move selected data points down
          D: Delete selected data points
          C: Clone selected data points
          I: Insert new data point of selected class
          P: Print the highlighted case details to the console
          ?: Infers the class of the highlighted case(s).
          G: Generate a specified number of data points (currently completely random)
          R: Relabel the selected data points with a selected class
            
        Mouse Controls for Plot Interaction:
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
        layout.addWidget(label)
        button = QPushButton("Close", clicked=self.accept)
        layout.addWidget(button)
        
        self.setLayout(layout)
