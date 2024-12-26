from PyQt6.QtWidgets import QMessageBox


def no_data_warning():
    no_data_message = QMessageBox()
    no_data_message.setWindowTitle('Warning: No dataset')
    no_data_message.setText('Please upload a dataset before generating the plot.')
    no_data_message.setIcon(QMessageBox.Icon.Warning)
    no_data_message.exec()


def odd_feature_count():
    no_data_message = QMessageBox()
    no_data_message.setWindowTitle('Warning: Odd Feature Count')
    no_data_message.setText('This plot type requires data with an even number of features.')
    no_data_message.setIcon(QMessageBox.Icon.Warning)
    no_data_message.exec()
