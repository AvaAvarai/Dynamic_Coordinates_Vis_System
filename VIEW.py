from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QColorDialog
from PyQt6.uic.load_ui import loadUi
import numpy as np
from PyQt6.QtWidgets import QApplication
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from ui import CLASS_TABLE, ATTRIBUTE_TABLE, PLOT
from utils import CLIPPING, WARNINGS

class View(QtWidgets.QMainWindow):
    def __init__(self, controller=None):
        super(View, self).__init__()

        self.controller: controller = controller
        loadUi('ui/GUI.ui', self)  # load .ui file for GUI made in Qt Designer

        self.plot_widget = None
        self.class_table = None
        self.attribute_table = None
        self.class_pl_exists = True
        self.attribute_pl_exists = True
        self.rule_count = 0
        self.cell_swap = QtWidgets.QTableWidget()
        self.plot_layout = self.findChild(QtWidgets.QVBoxLayout, 'plotDisplay')

        # Setup context menu for rulesListWidget
        self.rulesListWidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.rulesListWidget.customContextMenuRequested.connect(self.openContextMenu)
        
        self.rulesListWidget.itemClicked.connect(self.highlightAssociatedRegions)
    
        self.center_on_screen()
    
    def center_on_screen(self):
        '''Centers the window on the screen.'''
        screen = QApplication.primaryScreen()
        resolution = screen.geometry()
        self.move(int(resolution.width() / 2 - self.frameSize().width() / 2),             # win_width  <- screen_width  / 2 - frame_width / 2
                  int(resolution.height() / 2 - (self.frameSize().height() * 1.08) / 2))  # win_height <- screen_height / 2 - (frame_height * 1.08) / 2
    
    def highlightAssociatedRegions(self, item):
        rule_num = item.text().split()[1][:-1]
        rule_num = int(rule_num) - 1
        rules = self.controller.data.rule_regions
        if rule_num in rules:
            rule = rules[rule_num]
            primary_class, rects = rule
            
            # Check if it's already highlighted or not
            if not str(primary_class).endswith(" (highlighted)"):
                # Mark as highlighted
                new_primary_class = str(primary_class) + " (highlighted)"
            else:
                # Remove "(highlighted)" to toggle off the highlight
                new_primary_class = primary_class[:-14]
            
            # Update the rule with the new primary class name
            self.controller.data.rule_regions[rule_num] = (new_primary_class, rects)
            self.plot_widget.update()
                
    def removeSelectedRule(self):
        currentItem = self.rulesListWidget.currentItem()
        if currentItem:
            row = self.rulesListWidget.row(currentItem)
            item = self.rulesListWidget.takeItem(row)

            item_text = item.text()
            item_num = int(item_text.split()[1][:-1]) - 1

            if not item_num in self.controller.data.rule_regions:
                return
            
            self.controller.data.rule_regions.pop(item_num)

            
            self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
            # clip remaining rules and update clear_samples
            for rule in self.controller.data.rule_regions.values():
                for rect in rule[1]:
                    positions = self.controller.data.positions
                    CLIPPING.Clipping(rect, self.controller.data)
                    CLIPPING.clip_samples(positions, rect, self.controller.data)
            self.controller.data.clear_samples = np.add(self.controller.data.clear_samples, self.controller.data.clipped_samples)
            self.rule_count -= 1
            del item
            self.plot_widget.update()
            
    def openContextMenu(self, position):
        menu = QtWidgets.QMenu()
        removeAction = menu.addAction("Remove Rule")
        action = menu.exec(self.rulesListWidget.mapToGlobal(position))
        if action == removeAction:
            self.removeSelectedRule()

    def recenter_plot(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return

        self.plot_widget.reset_zoom()
        self.plot_widget.resize()
        
        self.refresh()

    def attr_slider(self):
        if not self.controller.data or not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        value = self.attribute_slide.value()
        self.controller.data.attribute_alpha = value
        self.plot_widget.update()

    def check_all_attr(self):
        if not self.controller.data:
            WARNINGS.no_data_warning()
            return
        ATTRIBUTE_TABLE.reset_checkmarks(self.attribute_table, self.controller.data.vertex_count, self.controller.data.plot_type)

    def check_all_class(self):
        if not self.controller.data:
            WARNINGS.no_data_warning()
            return
        CLASS_TABLE.reset_checkmarks(self.class_table, self.controller.data.class_count)

    def uncheck_all_attr(self):
        if not self.controller.data:
            WARNINGS.no_data_warning()
            return
        ATTRIBUTE_TABLE.uncheck_checkmarks(self.attribute_table, self.controller.data.vertex_count, self.controller.data.plot_type)

    def uncheck_all_class(self):
        if not self.controller.data:
            WARNINGS.no_data_warning()
            return
        CLASS_TABLE.uncheck_checkmarks(self.class_table, self.controller.data.class_count)

    def keyPressEvent(self, event) -> None:
        if self.plot_widget is None or self.controller.data is None:
            return

        key = event.key()

        if key == QtCore.Qt.Key.Key_Q:
            if self.controller.data.plot_type not in ['SCC', 'DCC']:
                self.controller.data.roll_clips(-1)
            else:
                self.controller.data.roll_vertex_in(-1)
            self.refresh()
        elif key == QtCore.Qt.Key.Key_E:
            if self.controller.data.plot_type not in ['SCC', 'DCC']:
                self.controller.data.roll_clips(1)
            else:
                self.controller.data.roll_vertex_in(1)
            self.refresh()
        elif key == QtCore.Qt.Key.Key_W:
            # move data samples up by 0.1 on all attributes and replot
            self.controller.data.move_samples(0.01)
            back_color = self.controller.view.plot_widget.background_color
            axes_color = self.controller.view.plot_widget.axes_color
            self.create_plot()
            self.controller.view.plot_widget.background_color = back_color
            self.controller.view.plot_widget.axes_color = axes_color
        elif key == QtCore.Qt.Key.Key_S:
            self.controller.data.move_samples(-0.01)
            back_color = self.controller.view.plot_widget.background_color
            axes_color = self.controller.view.plot_widget.axes_color
            self.create_plot()
            self.controller.view.plot_widget.background_color = back_color
            self.controller.view.plot_widget.axes_color = axes_color
        elif key == QtCore.Qt.Key.Key_P:
            # print dataframe information for clipped indices
            clipped_samples_bool = np.array(self.controller.data.clipped_samples, dtype=bool)
            print(self.controller.data.dataframe.loc[clipped_samples_bool])
            # and not normalized frame
            print(self.controller.data.not_normalized_frame.loc[clipped_samples_bool])
        elif key == QtCore.Qt.Key.Key_C:
            back_color = self.controller.view.plot_widget.background_color
            axes_color = self.controller.view.plot_widget.axes_color
            self.controller.data.copy_clip()
            self.controller.display_data()
            if self.plot_widget:
                self.plot_layout.removeWidget(self.plot_widget)
            self.plot_widget = PLOT.Plot(self.controller.data, self.highlight_overlaps_toggle, self.overlaps_textbox, self.controller.view.replot_overlaps_btn, parent=self)
            self.plot_layout.addWidget(self.plot_widget)
            self.controller.view.plot_widget.background_color = back_color
            self.controller.view.plot_widget.axes_color = axes_color
        elif key == QtCore.Qt.Key.Key_D:
            # delete all clipped samples from dataset
            self.controller.data.delete_clip()
            self.controller.display_data()
            back_color = self.controller.view.plot_widget.background_color
            axes_color = self.controller.view.plot_widget.axes_color
            if self.plot_widget:
                self.plot_layout.removeWidget(self.plot_widget)
            self.plot_widget = PLOT.Plot(self.controller.data, self.highlight_overlaps_toggle, self.overlaps_textbox, self.controller.view.replot_overlaps_btn, parent=self, reset_zoom=[self.plot_widget.m_left, self.plot_widget.m_right, self.plot_widget.m_bottom, self.plot_widget.m_top])
            self.plot_layout.addWidget(self.plot_widget)
            self.controller.view.plot_widget.background_color = back_color
            self.controller.view.plot_widget.axes_color = axes_color
        elif key == QtCore.Qt.Key.Key_I:
            # inject a data point with a value of 0.5 for each attribute show option to pick class
            # using function def inject_datapoint(self, data_point: List[float], class_name: str):
            data_point = [0.5] * self.controller.data.attribute_count
            class_name = None
            # show drop down with class names to select class_name
            class_name = QtWidgets.QInputDialog.getItem(self, "Select Class", "Select Class", self.controller.data.class_names, 0, False)
            self.controller.data.inject_datapoint(data_point, class_name[0])
            
            self.controller.display_data()
            if self.plot_widget:
                self.plot_layout.removeWidget(self.plot_widget)
            self.plot_widget = PLOT.Plot(self.controller.data, self.highlight_overlaps_toggle, self.overlaps_textbox, self.controller.view.replot_overlaps_btn, parent=self)
            self.plot_layout.addWidget(self.plot_widget)
        elif key == QtCore.Qt.Key.Key_G:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Generate Data")
            layout = QtWidgets.QVBoxLayout()
            dialog.setLayout(layout)

            num_samples_label = QtWidgets.QLabel("Enter the number of samples to generate:")
            layout.addWidget(num_samples_label)
            num_samples_spinbox = QtWidgets.QSpinBox()
            num_samples_spinbox.setMinimum(1)
            num_samples_spinbox.setMaximum(1000000)
            num_samples_spinbox.setValue(100)
            layout.addWidget(num_samples_spinbox)

            epochs_label = QtWidgets.QLabel("Enter the number of epochs to train CTGAN:")
            layout.addWidget(epochs_label)
            epochs_spinbox = QtWidgets.QSpinBox()
            epochs_spinbox.setMinimum(1)
            epochs_spinbox.setMaximum(1000000)
            epochs_spinbox.setValue(1000)
            layout.addWidget(epochs_spinbox)

            retain_data_checkbox = QtWidgets.QCheckBox("Retain previous data")
            layout.addWidget(retain_data_checkbox)

            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            layout.addWidget(buttons)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.controller.data.generate_data(num_samples_spinbox.value(), epochs_spinbox.value(), retain_data_checkbox.isChecked())
            else:
                return
            self.controller.display_data()
            back_color = self.controller.view.plot_widget.background_color
            axes_color = self.controller.view.plot_widget.axes_color
            if self.plot_widget:
                self.plot_layout.removeWidget(self.plot_widget)
            self.plot_widget = PLOT.Plot(self.controller.data, self.highlight_overlaps_toggle, self.overlaps_textbox, self.controller.view.replot_overlaps_btn, parent=self)
            self.plot_layout.addWidget(self.plot_widget)
            self.controller.view.plot_widget.background_color = back_color
            self.controller.view.plot_widget.axes_color = axes_color
        
        elif key == QtCore.Qt.Key.Key_R:
            # relabel the selected samples with a selected class
            class_name = QtWidgets.QInputDialog.getItem(self, "Select Class", "Select Class", self.controller.data.class_names, 0, False)
            self.controller.data.relabel_samples(class_name[0])
            self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
            self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
            
            self.refresh()
            self.controller.display_data()
            self.create_plot()
            
        elif key == QtCore.Qt.Key.Key_Question:
            # Only proceed if we have clipped samples
            if not np.any(self.controller.data.clipped_samples):
                WARNINGS.warning_message("No samples selected", "Please select samples using clipping before inferring classes.")
                return
                
            # Convert clipped_samples list to numpy array
            clipped_samples_array = np.array(self.controller.data.clipped_samples)
            
            # Get training data (all unclipped samples)
            train_mask = ~clipped_samples_array.astype(bool)
            X_train = self.controller.data.dataframe.iloc[train_mask].drop('class', axis=1)
            y_train = self.controller.data.dataframe.iloc[train_mask]['class']
            
            # Get samples to predict (clipped samples)
            test_mask = clipped_samples_array.astype(bool)
            X_test = self.controller.data.dataframe.iloc[test_mask].drop('class', axis=1)
            
            # Initialize classifiers
            classifiers = {
                'DT': DecisionTreeClassifier(random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=3),
                'SVM': SVC(kernel='rbf', probability=True),
                'Naive Bayes': GaussianNB(),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
            }
            
            # Train and predict with each classifier
            results = {}
            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                probabilities = clf.predict_proba(X_test)
                results[name] = {'predictions': predictions, 'probabilities': probabilities}
            
            # Create and show results table
            self.show_inference_results(results, X_test.index)
            
    def show_inference_results(self, results, sample_indices):
        """Display the inference results in a table."""
        # Create table window
        table_window = QtWidgets.QDialog(self)
        table_window.setWindowTitle("Class Inference Results")
        table_window.resize(1200, 600)  # Increased size to accommodate more columns
        
        # Create table
        table = QTableWidget()
        classifier_names = [
            'DT', 'KNN', 'SVM', 'Naive Bayes', 'Random Forest', 'AdaBoost',
            'Gradient Boosting', 'Extra Trees'
        ]
        table.setColumnCount(len(classifier_names) + 1)  # +1 for Sample ID
        table.setRowCount(len(sample_indices))
        
        # Set headers
        headers = ['Sample ID'] + classifier_names
        table.setHorizontalHeaderLabels(headers)
        
        # Fill table
        for row, idx in enumerate(sample_indices):
            # Sample ID
            table.setItem(row, 0, QTableWidgetItem(str(idx)))
            
            # Predictions from each classifier
            for col, classifier_name in enumerate(classifier_names, start=1):
                prediction = results[classifier_name]['predictions'][row]
                probs = results[classifier_name]['probabilities'][row]
                max_prob = max(probs)
                text = f"{prediction} ({max_prob:.2f})"
                item = QTableWidgetItem(text)
                table.setItem(row, col, item)
        
        # Adjust column widths
        table.resizeColumnsToContents()
        
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(table)
        
        # Add a label showing prediction agreement
        agreement_label = QtWidgets.QLabel()
        layout.addWidget(agreement_label)
        
        # Calculate and show prediction agreement
        for row in range(len(sample_indices)):
            predictions = [results[clf]['predictions'][row] for clf in classifier_names]
            unique_predictions = set(predictions)
            if len(unique_predictions) == 1:
                agreement_label.setText(f"All classifiers agree on predictions!")
            else:
                most_common = max(set(predictions), key=predictions.count)
                agreement_count = predictions.count(most_common)
                agreement_label.setText(
                    f"Most common prediction appears in {agreement_count}/{len(classifier_names)} classifiers"
                )
        
        # Add close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(table_window.close)
        layout.addWidget(close_button)
        
        table_window.setLayout(layout)
        table_window.exec()

    # function to refresh plot
    def refresh(self):
        if self.plot_widget:
            self.plot_widget.update()

    def axes_func(self):
        if not self.controller.data:
            WARNINGS.no_data_warning()
            return

        if self.show_axes.isChecked():
            self.controller.data.axis_on = True
        else:
            self.controller.data.axis_on = False

        self.refresh()
        
    def refresh_plot(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.vertex_in = np.zeros(self.controller.data.sample_count)
        self.controller.data.last_vertex_in = np.zeros(self.controller.data.sample_count)
        self.rulesListWidget.clear()
        self.controller.data.rule_regions = {}
        self.rule_count = 0
        self.controller.data.rule_count = 0
        self.controller.data.overlap_count = 0
        self.controller.data.reload()
        self.controller.display_data()
        self.create_plot()

    def create_plot(self):
        # Check if data not loaded
        if not (hasattr(self, 'controller') and self.controller is not None and self.controller.data is not None):
            WARNINGS.no_data_warning()
            return
        # Check if data load was cancelled
        if not (self.controller.data and self.controller.data.class_count > 0):
            print("No class data available to display in ClassTable.")
            return

        background_color = None
        axes_color = None
        # if plot widget exists, save the current background color and axes color
        if self.plot_widget:
            background_color = self.plot_widget.background_color
            axes_color = self.plot_widget.axes_color

        # remove initial placeholder
        if self.pl:
            self.plot_layout.removeWidget(self.pl)
        if self.plot_widget:
            self.plot_layout.removeWidget(self.plot_widget)

        self.controller.data.positions = []

        selected_plot_type = self.plot_select.currentText()

        if selected_plot_type == 'Parallel Coordinates':
            self.controller.data.plot_type = 'PC'
        elif selected_plot_type == 'Dynamic Scaffold Coordinates 1':
            self.controller.data.plot_type = 'DSC1'
        elif selected_plot_type == 'Dynamic Scaffold Coordinates 2':
            self.controller.data.plot_type = 'DSC2'
        elif selected_plot_type == 'Shifted Paired Coordinates':
            self.controller.data.plot_type = 'SPC'
        elif selected_plot_type == 'Static Circular Coordinates':
            self.controller.data.plot_type = 'SCC'
        elif selected_plot_type == 'Dynamic Circular Coordinates':
            self.controller.data.plot_type = 'DCC'
        else:
            return

        self.plot_widget = PLOT.Plot(self.controller.data, self.highlight_overlaps_toggle, self.overlaps_textbox, self.controller.view.replot_overlaps_btn, parent=self)
        
        # class table placeholder
        if self.class_pl_exists:
            self.class_table_layout.removeWidget(self.class_pl)
            self.class_table_layout.addWidget(self.class_table)
            self.class_pl_exists = False

        # attribute table placeholder
        if self.attribute_pl_exists:
            self.attribute_table_layout.removeWidget(self.attribute_pl)
            self.attribute_pl_exists = False
        else:
            self.attribute_table_layout.removeWidget(self.attribute_table)

        self.attribute_table = ATTRIBUTE_TABLE.AttributeTable(self.controller.data, self.replot_attributes, parent=self)
        self.attribute_table_layout.addWidget(self.attribute_table)

        self.plot_layout.addWidget(self.plot_widget)
        
        if self.class_table:
            self.class_table_layout.removeWidget(self.class_table)
        
        self.controller.view.class_table = CLASS_TABLE.ClassTable(self.controller.data, parent=self)
        self.class_table_layout.addWidget(self.controller.view.class_table)
        
        if background_color and axes_color:
            self.plot_widget.background_color = background_color
            self.plot_widget.axes_color = axes_color
        
    def analyze_clip(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        
        if len(self.plot_widget.all_rect) == 0:
            self.clipped_area_textbox.setText('No clipping area selected.')
            return
        CLIPPING.clip_files(self.controller.data, self.clipped_area_textbox)

    def undo_clip(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return

        # Ensure there is at least one rectangle to remove
        if self.plot_widget.all_rect:
            # Remove the last rectangle
            self.plot_widget.all_rect.pop()

            # Reset the clipping-related attributes before recalculating
            self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
            self.controller.data.vertex_in = np.zeros(self.controller.data.sample_count)
            self.controller.data.last_vertex_in = np.zeros(self.controller.data.sample_count)

            # Recalculate clipping for the remaining rectangles
            for rect in self.plot_widget.all_rect:
                positions = self.controller.data.positions
                CLIPPING.Clipping(rect, self.controller.data)
                CLIPPING.clip_samples(positions, rect, self.controller.data)

            # Update rule count if necessary
            if self.rule_count > 0:
                self.rule_count -= 1

            self.clipped_area_textbox.clear()

            self.plot_widget.update()
        else:
            print("No rectangles to remove.")

    def remove_clips(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return

        self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.vertex_in = np.zeros(self.controller.data.sample_count)
        self.controller.data.last_vertex_in = np.zeros(self.controller.data.sample_count)

        self.plot_widget.all_rect = []

        self.clipped_area_textbox.setText('')

        self.plot_widget.update()
        
        return

    def hide_clip(self):
        if self.controller.data.plot_type not in ['SCC', 'DCC']:
            self.controller.data.clear_samples = np.add(self.controller.data.clear_samples, self.controller.data.clipped_samples)
            self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        else:
            self.controller.data.clear_samples = np.add(self.controller.data.clear_samples, self.controller.data.vertex_in)
            self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)

    def remove_rules(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        
        self.rule_count = 0
        
        self.controller.data.rule_regions = {}
        self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
        
        self.rulesListWidget.clear()
        
        self.plot_widget.update()

    def trace_mode_func(self):
        self.controller.data.trace_mode = not self.controller.data.trace_mode
        self.plot_widget.update()

    def add_rule(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        
        if not self.plot_widget.all_rect:
            print("No clipping area selected.")
            return

        rules = self.plot_widget.all_rect
        
        skips = []
        for i, _ in enumerate(self.controller.data.clear_samples):
            if self.controller.data.clear_samples[i] == 1:
                skips.append(i)
        class_set = CLIPPING.count_clipped_classes(self.controller.data, skips)
        if len(class_set) == 1:
            class_add = class_set.pop()
            primary_class = class_add  + " (pure)"
            class_set.add(class_add)
        else:
            primary_class = CLIPPING.primary_clipped_class(self.controller.data)
        if primary_class in self.controller.data.rule_regions:
            primary_class = str(primary_class) + f" ({str(self.controller.data.rule_count)})"
        else:
            self.controller.data.rule_regions[self.rule_count] = (str(primary_class), rules)
        class_str = ""
        for index, c in enumerate(class_set):
            class_str += str(c)
            if index < len(class_set) - 1:
                class_str += ", "
        
        overcounts = np.count_nonzero(np.logical_and(self.controller.data.clipped_samples, self.controller.data.clear_samples))
        
        case_count = CLIPPING.count_clipped_samples(self.controller.data)

        case_count -= overcounts
        class_count = len(class_set)

        if class_count == 1:
            class_str += " class"
        else:
            class_str += " classes"
        
        cases_str = ""
        if case_count == 1:
            cases_str = "case"
        else:
            cases_str = "cases"
        
        region_str = ""
        if len(rules) == 1:
            region_str = "region"
        else:
            region_str = "regions"
        
        rule_description = f"Rule {self.rule_count + 1}) {case_count} {cases_str} {class_str} {len(rules)} {region_str}"
        
        item = QtWidgets.QListWidgetItem(rule_description)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.CheckState.Checked)

        self.rulesListWidget.addItem(item)
        self.rulesListWidget.itemChanged.connect(self.onRuleItemChanged)

        self.rule_count += 1
        self.plot_widget.all_rect = []
        self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        
        # check all the rule boxes to show the rules
        for i in range(self.rulesListWidget.count()):
            self.rulesListWidget.item(i).setCheckState(QtCore.Qt.CheckState.Checked)
        
        self.plot_widget.update()

    def onRuleItemChanged(self, item):
        try:
            # Extract rule number from item text
            rule_num = item.text().split()[1][:-1]
            rule_num = int(rule_num) - 1

            # Ensure the rule_num is within the range of existing rules
            if rule_num < 0 or rule_num >= len(self.controller.data.rule_regions):
                print(f"Rule number {rule_num} is out of range.")
                return

            rules = self.controller.data.rule_regions
            rule_keys = list(rules.keys())
            
            # Ensure the rule_key is valid
            if rule_num < len(rule_keys):
                rule = rules[rule_keys[rule_num]]

                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    for rect in rule[1]:
                        positions = self.controller.data.positions
                        CLIPPING.Clipping(rect, self.controller.data)
                        CLIPPING.clip_samples(positions, rect, self.controller.data)

                    self.controller.data.clear_samples = np.subtract(self.controller.data.clear_samples, self.controller.data.clipped_samples)
                    self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
                else:
                    for rect in rule[1]:
                        positions = self.controller.data.positions
                        CLIPPING.Clipping(rect, self.controller.data)
                        CLIPPING.clip_samples(positions, rect, self.controller.data)
                    self.controller.data.clear_samples = np.add(self.controller.data.clear_samples, self.controller.data.clipped_samples)
                    self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
                    
                self.plot_widget.update()
            else:
                print(f"Rule key {rule_num} is out of range.")
        except Exception as e:
            print(f"Error in onRuleItemChanged: {e}")

    def table_swap(self, event):
        table = event.source()

        if table == self.class_table:
            CLASS_TABLE.table_swap(table, self.controller.data, self.plot_widget, event)
        elif table == self.attribute_table:
            ATTRIBUTE_TABLE.table_swap(table, self.controller.data, event, self.replot_attributes)

        event.accept()

    def replot_attributes(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        if self.controller.data.dataframe is None:
            return
        self.controller.data.attribute_names.append('class')
        self.controller.data.dataframe = self.controller.data.dataframe[self.controller.data.attribute_names]

        self.controller.data.attribute_names.pop()
        self.controller.data.positions = []
        self.controller.data.active_attributes = np.repeat(True, self.controller.data.attribute_count)
        ATTRIBUTE_TABLE.reset_checkmarks(self.attribute_table, self.controller.data.vertex_count, self.controller.data.plot_type)
        if self.attribute_table:
            self.attribute_table.clearTableWidgets()
        v1, v2, v3, v4 = self.plot_widget.get_zoom()
        self.create_plot()
        self.plot_widget.set_zoom(v1, v2, v3, v4)

    def open_background_color_picker(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.background_color = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
            self.plot_widget.redraw_plot(background_color=self.background_color)

    def open_axes_color_picker(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.axes_color = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
            self.plot_widget.redraw_plot(axes_color=self.axes_color)

    def replot_overlaps(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        self.plot_widget.replot_overlaps()

    def highlight_overlaps(self):
        if not self.plot_widget:
            WARNINGS.no_data_warning()
            return
        self.plot_widget.highlight_overlaps = not self.plot_widget.highlight_overlaps
        self.plot_widget.update()
