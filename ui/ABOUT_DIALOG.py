from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Information")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        header = QLabel("About DCVis")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        about_info = """
        This project is a Visual Knowledge Discovery system using the lossless multidimensional General Line Coordinates (GLC)
        for building visual machine learning models, generate synthetic data, and edit data characteristics by data domain expert.
        
        Key Features:
          • Multidimensional data visualization
          • Data classification
          • Synthetic data generation
        
        Please see the README.md file for more information.
        
        If you encounter any problems please let us known through our Github issues page, thank you!
        
        DCVis, the Dynamic Coordinates Visualisation System, rebuilt from DSCVis, the Dynamic Scaffold Coordinates Visualisation System,
        This project is an ongoing research project at Central Washington University's Visual Knowledge Discovery and Imaging Lab.
        """
        
        label = QLabel(about_info)
        label.setStyleSheet("font-size: 14px;")
        layout.addWidget(label)
        button = QPushButton("Close", clicked=self.accept)
        layout.addWidget(button)
        
        self.setLayout(layout)
