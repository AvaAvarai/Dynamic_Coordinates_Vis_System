from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt


def table_swap(table, dataset, event, replot_func):
    moved_from = table.currentRow()
    from_item = table.item(moved_from, 0)
    if from_item is None:  # if no item is selected
        return
    from_item = from_item.text()
    moved_to = table.rowAt(round(event.position().y()))

    to_item = table.item(moved_to, 0)
    if to_item is None:  # if no item is selected
        return
    to_item = to_item.text()

    table.item(moved_from, 0).setText(to_item)
    table.item(moved_to, 0).setText(from_item)
    
    place_holder = dataset.attribute_names[moved_from]
    dataset.attribute_names[moved_from] = dataset.attribute_names[moved_to]
    dataset.attribute_names[moved_to] = place_holder

    place_holder = dataset.attribute_order[moved_from]
    dataset.attribute_order[moved_from] = dataset.attribute_order[moved_to]
    dataset.attribute_order[moved_to] = place_holder

    place_holder = dataset.active_attributes[moved_from]
    dataset.active_attributes[moved_from] = dataset.active_attributes[moved_to]
    dataset.active_attributes[moved_to] = place_holder

    if dataset.plot_type == 'DCC':
        dataset.coefs[moved_from], dataset.coefs[moved_to] = dataset.coefs[moved_to], dataset.coefs[moved_from]

    if dataset.plot_type == 'PC':
        # Swap the vertical shifts as well
        place_holder = dataset.axis_vertical_shifts[moved_from]
        dataset.axis_vertical_shifts[moved_from] = dataset.axis_vertical_shifts[moved_to]
        dataset.axis_vertical_shifts[moved_to] = place_holder

    replot_func()


def reset_checkmarks(table, count, plot_type):
    for idx in range(count):
        if plot_type in ['SPC', 'DSC2']:  # paired plots
            idx *= 2
        cell = table.cellWidget(idx, 1)
        if isinstance(cell, QtWidgets.QCheckBox):  # Check if the widget is a QCheckBox
            cell.setCheckState(Qt.CheckState.Checked)


def uncheck_checkmarks(table, count, plot_type):
    for idx in range(count):
        if plot_type in ['SPC', 'DSC2']:  # paired plots
            idx *= 2
        cell = table.cellWidget(idx, 1)
        if isinstance(cell, QtWidgets.QCheckBox):  # Check if the widget is a QCheckBox
            cell.setCheckState(Qt.CheckState.Unchecked)


class AttributeTable(QtWidgets.QTableWidget):
    def __init__(self, dataset, replot_func, parent=None):
        super(AttributeTable, self).__init__(parent)

        self.dataset = dataset
        self.replot_func = replot_func

        if self.dataset.plot_type == 'PC':
            self.setColumnCount(4)
            self.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Order'))
            self.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Alpha'))
            self.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Invert'))
            self.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('Position'))
        elif self.dataset.plot_type == 'DCC':
            self.setColumnCount(4)
            self.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Coefficient'))
            self.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('Value'))
        else:
            self.setColumnCount(3)
            self.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Invert'))
        self.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Order'))
        self.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Alpha'))
            
        self.setRowCount(self.dataset.attribute_count)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        if not self.dataset.plot_type == 'DCC':
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)    
        else:
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)

        if self.dataset.plot_type == 'PC':
            for idx in range(self.dataset.attribute_count):
                self.init_pc_row(idx)
        elif self.dataset.plot_type == 'DCC':
            for idx in range(self.dataset.attribute_count):
                self.init_dcc_row(idx, self.dataset.attribute_count)
        for idx, attribute_name in enumerate(self.dataset.attribute_names):
            self.setItem(idx, 0, QtWidgets.QTableWidgetItem(attribute_name))
            self.setCellWidget(idx, 1, CheckBox(idx, self.dataset, 'Alpha', parent=self))
            if not self.dataset.plot_type == 'DCC':    
                self.setCellWidget(idx, 2, InversionCheckBox(idx, self.dataset, self.replot_func, parent=self))

    def init_dcc_row(self, idx, att_count):
        # Initialize slider
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(-100 * att_count)  # [-1, 1] range for coefficients
        slider.setMaximum(100 * att_count)
        slider.setValue(int(self.dataset.coefs[idx]))
        slider.valueChanged.connect(lambda value, x=idx: self.update_text_box(x, value, update_dataset=True))

        # Initialize textbox
        textbox = QtWidgets.QLineEdit()
        textbox.setText(str(int(self.dataset.coefs[idx])))
        textbox.setValidator(QtGui.QIntValidator(0, 100))
        textbox.textChanged.connect(lambda value, x=idx: self.update_slider(x, value, update_dataset=True))

        self.setCellWidget(idx, 2, slider)
        self.setCellWidget(idx, 3, textbox)

    def update_slider(self, idx, value, update_dataset=False):
        if value:
            self.cellWidget(idx, 2).setValue(int(value))
            if update_dataset:
                self.dataset.update_coef(idx, int(value))
                self.replot_func()

    def update_text_box(self, idx, value, update_dataset=False):
        if self.cellWidget(idx, 3) is None:
            return
        self.cellWidget(idx, 3).setText(str(value))
        if update_dataset:
            self.dataset.update_coef(idx, int(value))
            self.replot_func()
    
    def clearTableWidgets(self):
        # Loop through each row and column to remove widgets and clear items
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                widget = self.cellWidget(row, col)
                if widget is not None:
                    # Remove the widget from the cell
                    self.removeCellWidget(row, col)
                    widget.deleteLater()  # Ensure the widget is deleted properly

        # Clear all the contents of the table after removing widgets
        self.clearContents()

    def init_pc_row(self, idx):
        # Initialize vertical position slider
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(int(self.dataset.axis_vertical_shifts[idx] * 100))  # Set initial value
        
        # Create a widget to hold the slider
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        
        # Add a label to show the current value
        value_label = QtWidgets.QLabel(f"{slider.value()}")
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        value_label.setMinimumWidth(30)
        
        def on_value_changed(value):
            value_label.setText(f"{value}")
            self.update_axis_position(idx, value)
            
        slider.valueChanged.connect(on_value_changed)
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.setCellWidget(idx, 3, container)

    def update_axis_position(self, idx, value):
        # Update the axis vertical position
        self.dataset.axis_vertical_shifts[idx] = value / 100  # Convert to range [-1, 1]
        self.replot_func()


class CheckBox(QtWidgets.QCheckBox):
    def __init__(self, row, dataset, option, parent=None):
        super(CheckBox, self).__init__(parent)
        self.index = row
        self.data = dataset
        self.option = option
        self.setCheckState(Qt.CheckState.Checked)
        self.stateChanged.connect(self.show_hide_classes)
        self.setStyleSheet("margin-left: 12px;")

    def show_hide_classes(self):
        if self.isChecked():
            if self.data.plot_type == 'SPC' or self.data.plot_type == 'DSC2':
                self.data.active_attributes[int(self.index / 2)] = True
            else:
                self.data.active_attributes[self.index] = True
        else:
            if self.data.plot_type == 'SPC' or self.data.plot_type == 'DSC2':
                self.data.active_attributes[int(self.index / 2)] = False
            else:
                self.data.active_attributes[self.index] = False


class InversionCheckBox(QtWidgets.QCheckBox):
    def __init__(self, index, dataset, replot_func, parent=None):
        super(InversionCheckBox, self).__init__(parent)
        self.index = index
        self.dataset = dataset
        self.replot_func = replot_func
        self.setStyleSheet("margin-left: 12px;")

        # Initialize the checkbox state based on the attribute_inversions list
        self.setChecked(self.dataset.attribute_inversions[self.index])

        # Connect the stateChanged signal to the toggle_inversion method
        self.stateChanged.connect(self.toggle_inversion)

    def toggle_inversion(self):
        self.dataset.attribute_inversions[self.index] = not self.dataset.attribute_inversions[self.index]
        self.replot_func()

