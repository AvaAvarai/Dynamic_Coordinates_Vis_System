from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QColorDialog
from PyQt6.uic.load_ui import loadUi

import numpy as np
import sys

import CLASS_TABLE
import ATTRIBUTE_TABLE
import PLOT
import CLIPPING
import WARNINGS
import CONTROLLER

class UiView(QtWidgets.QMainWindow):
    def __init__(self, controller=None):
        super(UiView, self).__init__()
        
        self.controller: controller = controller
        loadUi('GUI.ui', self)  # load .ui file for GUI made in Qt Designer

        self.plot_widget = None

        self.class_table = None
        self.attribute_table = None
        
        self.class_pl_exists = True
        self.attribute_pl_exists = True

        # for swapping cells
        self.cell_swap = QtWidgets.QTableWidget()
        self.plot_layout = self.findChild(QtWidgets.QVBoxLayout, 'plotDisplay')

    def recenter_plot(self):
        if not self.plot_widget:
            WARNINGS.noDataWarning()
            return
        # for zooming
        self.plot_widget.m_left = -1.125
        self.plot_widget.m_right = 1.125
        self.plot_widget.m_bottom = -1.125
        self.plot_widget.m_top = 1.125
        
        if self.controller.data.plot_type in ['SCC', 'DCC']: # fit CC to window
            self.plot_widget.m_left = -self.controller.data.attribute_count * 0.35
            self.plot_widget.m_right = self.controller.data.attribute_count * 0.35
            self.plot_widget.m_bottom = -self.controller.data.attribute_count * 0.35
            self.plot_widget.m_top = self.controller.data.attribute_count * 0.35

        self.refresh()

    # function to get alpha value for hidden attributes
    def attr_slider(self):
        if not self.controller.data or not self.plot_widget:
            WARNINGS.noDataWarning()
            return
        value = self.attribute_slide.value()
        self.controller.data.attribute_alpha = value
        self.plot_widget.update()

    def check_all_attr(self):
        if not self.controller.data:
            WARNINGS.noDataWarning()
            return
        ATTRIBUTE_TABLE.reset_checkmarks(self.attribute_table, self.controller.data.vertex_count, self.controller.data.plot_type)

    def check_all_class(self):
        if not self.controller.data:
            WARNINGS.noDataWarning()
            return
        CLASS_TABLE.reset_checkmarks(self.class_table, self.controller.data.class_count)

    def uncheck_all_attr(self):
        if not self.controller.data:
            WARNINGS.noDataWarning()
            return
        ATTRIBUTE_TABLE.uncheck_checkmarks(self.attribute_table, self.controller.data.vertex_count, self.controller.data.plot_type)

    def uncheck_all_class(self):
        if not self.controller.data:
            WARNINGS.noDataWarning()
            return
        CLASS_TABLE.uncheck_checkmarks(self.class_table, self.controller.data.class_count)

    # function to refresh plot
    def refresh(self):
        if self.plot_widget:
            self.plot_widget.update()

    def axes_func(self):
        if not self.controller.data:
            WARNINGS.noDataWarning()
            return

        if self.show_axes.isChecked():
            self.controller.data.axis_on = True
        else:
            self.controller.data.axis_on = False

        self.refresh()

    def create_plot(self):
        if not self.controller.data:
            WARNINGS.noDataWarning()
            return

        # remove initial placeholder
        if self.pl:
            self.plot_layout.removeWidget(self.pl)
        if self.plot_widget:
            self.plot_layout.removeWidget(self.plot_widget)

        self.controller.data.positions = []
        self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.vertex_in = np.zeros(self.controller.data.sample_count)
        self.controller.data.last_vertex_in = np.zeros(self.controller.data.sample_count)
        
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
        
        self.plot_widget = PLOT.MakePlot(self.controller.data, parent=self)
        self.remove_clip() # remove previous clips if any
        
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

    # function to save clip files
    def test(self):
        if not self.plot_widget:
            WARNINGS.noDataWarning()
            return

        CLIPPING.clip_files(self.controller.data, self.clipped_area_textbox)

    def remove_clip(self):
        if not self.plot_widget:
            WARNINGS.noDataWarning()
            return
        
        self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.clear_samples = np.zeros(self.controller.data.sample_count)
        self.controller.data.vertex_in = np.zeros(self.controller.data.sample_count)
        self.controller.data.last_vertex_in = np.zeros(self.controller.data.sample_count)
        self.plot_widget.all_rect = []

        self.clipped_area_textbox.setText('')

        self.plot_widget.update()
    
    def hide_clip(self):
        self.controller.data.clear_samples = np.add(self.controller.data.clear_samples, self.controller.data.clipped_samples)
        self.controller.data.clipped_samples = np.zeros(self.controller.data.sample_count)
        
        self.plot_widget.update()

    def table_swap(self, event):
        table = event.source()

        if table == self.class_table:
            CLASS_TABLE.table_swap(table, self.controller.data, self.plot_widget, event)
        elif table == self.attribute_table:
            ATTRIBUTE_TABLE.table_swap(table, self.controller.data, event, self.replot_attributes)

        event.accept()

    def replot_attributes(self):
        if not self.plot_widget:
            WARNINGS.noDataWarning()
            return
        
        self.controller.data.attribute_names.append('class')
        self.controller.data.dataframe = self.controller.data.dataframe[self.controller.data.attribute_names]

        self.controller.data.attribute_names.pop()
        self.controller.data.positions = []
        self.controller.data.active_attributes = np.repeat(True, self.controller.data.attribute_count)
        ATTRIBUTE_TABLE.reset_checkmarks(self.attribute_table, self.controller.data.vertex_count, self.controller.data.plot_type)
        
        self.create_plot()

    def open_background_color_picker(self):
        if not self.plot_widget:
            WARNINGS.noDataWarning()
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.background_color = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
            self.plot_widget.redraw_plot(background_color=self.background_color)

    def open_axes_color_picker(self):
        if not self.plot_widget:
            WARNINGS.noDataWarning()
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.axes_color = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
            self.plot_widget.redraw_plot(axes_color=self.axes_color)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    view = UiView()
    
    controller = CONTROLLER.Controller(view)
    view.controller = controller

    view.show()
    app.exec()
