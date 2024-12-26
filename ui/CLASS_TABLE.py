from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush


def reset_checkmarks(table, count):
    for idx in range(count):
        cell1 = table.cellWidget(idx, 1)
        cell1.setCheckState(Qt.CheckState.Checked)
        cell2 = table.cellWidget(idx, 2)
        cell2.setCheckState(Qt.CheckState.Checked)

def uncheck_checkmarks(table, count):
    for idx in range(count):
        cell1 = table.cellWidget(idx, 1)
        cell1.setCheckState(Qt.CheckState.Unchecked)
        cell2 = table.cellWidget(idx, 2)
        cell2.setCheckState(Qt.CheckState.Unchecked)

def table_swap(table, dataset, plot, event):
    moved_from = table.currentRow()
    moved_to = table.rowAt(round(event.position().y()))

    if moved_from == moved_to or moved_to == -1:
        return

    # Swap the text and colors
    from_item = table.item(moved_from, 0)
    to_item = table.item(moved_to, 0)

    from_text = from_item.text()
    to_text = to_item.text()

    from_color = from_item.foreground().color()
    to_color = to_item.foreground().color()

    from_item.setText(to_text)
    from_item.setForeground(QBrush(to_color))

    to_item.setText(from_text)
    to_item.setForeground(QBrush(from_color))

    # Swap the class orders
    dataset.class_order[moved_from], dataset.class_order[moved_to] = dataset.class_order[moved_to], dataset.class_order[moved_from]

    plot.update()


class ClassTable(QtWidgets.QTableWidget):
    refresh_GUI = pyqtSignal()

    def __init__(self, dataset, parent=None):
        super(ClassTable, self).__init__(parent)

        self.data = dataset
        self.refresh_GUI.connect(self.parent().refresh)
        
        if not (self.data.plot_type == 'SCC' or self.data.plot_type == 'DCC'):
            self.setColumnCount(4)
            self.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('Color'))
        else:
            self.setColumnCount(5)
            self.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('Sector'))
            self.setHorizontalHeaderItem(4, QtWidgets.QTableWidgetItem('Color'))

        self.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Order'))
        self.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Lines'))
        self.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Points'))
        self.setRowCount(dataset.class_count)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        class_header = self.horizontalHeader()
        class_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        class_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)

        counter = 0
        for ele in dataset.class_names:
            class_name = QtWidgets.QTableWidgetItem(str(ele))
            class_name.setForeground(QBrush(QColor(dataset.class_colors[dataset.class_order[counter]][0], dataset.class_colors[dataset.class_order[counter]][1], dataset.class_colors[dataset.class_order[counter]][2])))
            self.setItem(counter, 0, class_name)

            class_checkbox = CheckBox(counter, dataset, self.refresh_GUI, 'class', parent=self)
            self.setCellWidget(counter, 1, class_checkbox)

            marker_checkbox = CheckBox(counter, dataset, self.refresh_GUI, 'marker', parent=self)
            self.setCellWidget(counter, 2, marker_checkbox)

            if dataset.plot_type == 'SCC' or dataset.plot_type == 'DCC':
                sector_checkbox = CheckBox(counter, dataset, self.refresh_GUI, 'sector', parent=self)
                self.setCellWidget(counter, 3, sector_checkbox)
                button = Button(counter, dataset, self.refresh_GUI, parent=self)
                self.setCellWidget(counter, 4, button)
            else:
                button = Button(counter, dataset, self.refresh_GUI, parent=self)
                self.setCellWidget(counter, 3, button)

            counter += 1


# class button for changing color
class Button(QtWidgets.QPushButton):
    def __init__(self, row, dataset, refresh, parent=None):
        super(Button, self).__init__(parent=parent)

        self.setText('Color')
        self.index = dataset.class_order[row]
        self.cell = self.parent().item(self.index, 0)
        self.data = dataset
        self.r = refresh
        self.setMaximumWidth(50)
        self.clicked.connect(self.color_dialog)

    def color_dialog(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            rgb = color.getRgb()
            self.cell.setForeground(QBrush(QColor(rgb[0], rgb[1], rgb[2])))
            self.data.class_colors[self.index] = [rgb[0], rgb[1], rgb[2]]


# class for checkbox in the class table
class CheckBox(QtWidgets.QCheckBox):
    def __init__(self, row, dataset, refresh, option, parent=None):
        super(CheckBox, self).__init__(parent)
        self.index = row
        self.data = dataset
        self.r = refresh
        self.option = option
        self.setCheckState(Qt.CheckState.Checked)
        self.stateChanged.connect(self.show_hide_classes)
        self.setStyleSheet("margin-left: 12px;")

    def show_hide_classes(self):
        if self.isChecked():
            if self.option == 'class':
                self.data.active_classes[self.index] = True
            elif self.option == 'marker':
                self.data.active_markers[self.index] = True
            elif self.option == 'sector':
                self.data.active_sectors[self.index] = True
        else:
            if self.option == 'class':
                self.data.active_classes[self.index] = False
            elif self.option == 'marker':
                self.data.active_markers[self.index] = False
            elif self.option == 'sector':
                self.data.active_sectors[self.index] = False

        self.r.emit()
