from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from ui import HELP_DIALOG, ABOUT_DIALOG, CLASS_TABLE

import sys

import MODEL

class Controller:
    def __init__(self, view, dataset=None):
        self.data = dataset
        self.view = view
        if self.view is not None:
            self.setup_connections()
            self.setup_menu()

    def setup_menu(self):
        self.view.menuHelp.addAction(self.view.actionControls_List)
        self.view.actionControls_List.triggered.connect(self.show_help_dialog)
        self.view.menuHelp.addAction(self.view.actionAbout)
        self.view.actionAbout.triggered.connect(self.show_about_dialog)

    def show_help_dialog(self):
        dialog = HELP_DIALOG.HelpDialog(self.view)
        dialog.exec()  # modal dialog

    def show_about_dialog(self):
        dialog = ABOUT_DIALOG.AboutDialog(self.view)
        dialog.exec()  # modal dialog

    def setup_connections(self):
        # Bind buttons to functions
        self.view.load_button.clicked.connect(self.load_dataset)
        self.view.plot_button.clicked.connect(self.view.create_plot)
        self.view.exit_button.clicked.connect(lambda: sys.exit())
        self.view.actionExit.triggered.connect(lambda: sys.exit())
        self.view.analyze_clips_btn.clicked.connect(self.view.analyze_clip)
        self.view.clear_last_clip_btn.clicked.connect(self.view.undo_clip)
        self.view.clear_all_clips_btn.clicked.connect(self.view.remove_clips)
        self.view.recenter_button.clicked.connect(self.view.recenter_plot)
        self.view.add_class_rule_btn.clicked.connect(self.view.add_rule)
        self.view.clear_class_rules_btn.clicked.connect(self.view.remove_rules)
        self.view.show_axes.stateChanged.connect(self.view.axes_func)
        self.view.attribute_slide.valueChanged.connect(self.view.attr_slider)
        self.view.check_classes.clicked.connect(self.view.check_all_class)
        self.view.uncheck_classes.clicked.connect(self.view.uncheck_all_class)
        self.view.check_attributes.clicked.connect(self.view.check_all_attr)
        self.view.uncheck_attributes.clicked.connect(self.view.uncheck_all_attr)
        self.view.cell_swap.__class__.dropEvent = self.view.table_swap
        self.view.background_button.clicked.connect(self.view.open_background_color_picker)
        self.view.axes_button.clicked.connect(self.view.open_axes_color_picker)
        self.view.trace_mode.clicked.connect(self.view.trace_mode_func)
        self.view.replot_overlaps_btn.clicked.connect(self.view.replot_overlaps)
        self.view.replot_overlaps_btn.setEnabled(False)
        self.view.save_model_button.clicked.connect(self.save_model)
        self.view.refresh_button.clicked.connect(self.view.refresh_plot)
        self.view.highlight_overlaps_toggle.clicked.connect(self.view.highlight_overlaps)
        self.view.highlight_overlaps_toggle.setEnabled(False)
        
        # Set keyboard shortcuts
        self.view.load_button.setShortcut(Qt.Key.Key_F1)
        self.view.plot_button.setShortcut(Qt.Key.Key_F3)
        self.view.refresh_button.setShortcut(Qt.Key.Key_F4)
        self.view.exit_button.setShortcut(Qt.Key.Key_Escape)
        self.view.recenter_button.setShortcut(Qt.Key.Key_F2)
        self.view.trace_mode.setShortcut(Qt.Key.Key_T)
        self.view.show_axes.setShortcut(Qt.Key.Key_A)

    def display_data(self):
        data_info_string = f'Dataset Name: {self.data.name} \nNumber of classes: {self.data.class_count} attributes: {self.data.attribute_count} samples: {self.data.sample_count}'

        for index, ele in enumerate(self.data.class_names):
            if len(self.data.count_per_class) > index:
                data_info_string += f'\nClass {index + 1}: {ele} sample count: {self.data.count_per_class[index]}'

        self.view.dataset_textbox.setText(data_info_string)

    def save_model(self):
        if not self.data or self.data.not_normalized_frame is None or self.data.not_normalized_frame.empty:
            QtWidgets.QMessageBox.warning(self.view, "Warning", "There is no data to save.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.view, "Save File", "", "CSV Files (*.csv)")
        if filename:
            try:
                # Ensure the filename has the correct extension
                if not filename.endswith('.csv'):
                    filename += '.csv'
                # Save the DataFrame to the specified CSV file
                self.data.not_normalized_frame.to_csv(filename, index=False)
                QtWidgets.QMessageBox.information(self.view, "Success", "The dataset has been saved successfully.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.view, "Error", f"An error occurred while saving the file: {e}")

    def load_dataset(self):
        if self.data:
            del self.data

        self.data = MODEL.Dataset()
        filename = QtWidgets.QFileDialog.getOpenFileName(self.view, 'Open File', 'datasets')
        if filename[0] == '':
            return

        # GUI changes for changing datasets without restarting the application
        if self.view.plot_widget and self.view.class_table and self.view.plot_layout:
            self.view.plot_layout.removeWidget(self.view.plot_widget)
            del self.view.plot_widget
            self.view.plot_widget = None
            self.view.plot_layout.addWidget(self.view.pl)

            self.view.class_table_layout.removeWidget(self.view.class_table)
            del self.view.class_table
            self.view.class_table = None
            self.view.class_table_layout.addWidget(self.view.class_pl)
            self.view.class_pl_exists = True

            self.view.attribute_table_layout.removeWidget(self.view.attribute_table)
            del self.view.attribute_table
            self.view.attribute_table = None
            self.view.attribute_table_layout.addWidget(self.view.attribute_pl)
            self.view.attribute_pl_exists = True

        self.data.load_from_csv(filename[0])
        self.display_data()
        self.view.class_table = CLASS_TABLE.ClassTable(self.data, parent=self.view)
