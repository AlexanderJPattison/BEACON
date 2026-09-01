import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from beacon_client import BEACON_Client

from matplotlib.backends.backend_qt5agg import FigureCanvas

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot

class WorkerSignals(QObject):
    status = pyqtSignal(bytes)
    figure = pyqtSignal(bytes)
    images = pyqtSignal(bytes)
    stopped = pyqtSignal(int)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
    
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signal = WorkerSignals()
        
        # Add the callback to our kwargs
        self.kwargs['status_callback'] = self.signal.status
        self.kwargs['figure_callback'] = self.signal.figure
        self.kwargs['images_callback'] = self.signal.images
        self.kwargs['stopped_callback'] = self.signal.stopped
    
    @pyqtSlot()
    def run(self):
        result = self.fn(*self.args, **self.kwargs)


class Widget(QWidget):
    def __init__(self, host, port, parent=None):
        '''
        Initializes the GUI
        '''
        super().__init__(parent)
        
        self.setWindowTitle('Bayesian-Enhanced Aberration Correction and Optimization Network (BEACON)')
        
        outerLayout = QHBoxLayout()
        
        controlPanelLayout = QVBoxLayout()
        
        abOptionsLayout = QGridLayout()
        abOptionsLayout.addWidget(QLabel('Aberrations'),0,0)
        abOptionsLayout.addWidget(QLabel('Lower Bound'),0,1)
        abOptionsLayout.addWidget(QLabel('Upper Bound'),0,2)
        abOptionsLayout.addWidget(QLabel('Select'),0,3)
        
        self.ab_list = ['C1','A1_x','A1_y','B2_x','B2_y','A2_x','A2_y']
        self.ab_display_list = ['C1','A1 (x)','A1 (y)','B2 (x)','B2 (y)','A2 (x)','A2 (y)']
        self.ab_default_ranges = ['600','10','10','300','300','300','300']
        
        self.check_boxes = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.fine_coarse = []
        
        for i in range(0,len(self.ab_list)):
            self.check_boxes.append(QCheckBox(f'{self.ab_display_list[i]}'))
            self.lower_bounds.append(QLineEdit(f'-{self.ab_default_ranges[i]}'))
            self.upper_bounds.append(QLineEdit(f'{self.ab_default_ranges[i]}'))
            
            abOptionsLayout.addWidget(self.check_boxes[i],i+1,0)
            abOptionsLayout.addWidget(self.lower_bounds[i],i+1,1)
            abOptionsLayout.addWidget(self.upper_bounds[i],i+1,2)
            
            fc_toggle = QComboBox()
            
            if self.ab_display_list[i] == 'C1' or self.ab_display_list[i] == 'C3':
                fc_toggle.addItems(['None'])
            else:
                fc_toggle.addItems(['coarse', 'fine'])
            self.fine_coarse.append(fc_toggle)
            abOptionsLayout.addWidget(self.fine_coarse[i],i+1,3)
        
        self.check_boxes[0].setChecked(True)
        #self.check_boxes[1].setChecked(True)
        #self.check_boxes[2].setChecked(True)
        
        self.dwell_input = QLineEdit('2')
        self.metric_input = QComboBox()
        self.metric_input.addItems(['Normalised Variance', 'Variance', 'Standard Deviation', 'Defocus Slice'])
        self.metric_input_names = ['normvar', 'var', 'std', 'df_slice']
        self.init_size_input = QLineEdit('5')
        self.iterations_input = QLineEdit('5')
        self.extra_iterations_input = QLineEdit('5')
        self.acq_func_input = QComboBox()
        self.acq_func_input.addItems(['Upper Confidence Bound'])
        self.acq_func_input_names = ['ucb']
        self.ucb_coefficient_input = QLineEdit('2.0')
        self.noise_level_input = QLineEdit('0.01')
        self.return_images_input = QCheckBox()
        self.return_images_input.setChecked(True)
        self.ccorr_input = QCheckBox()
        self.ccorr_input.setChecked(True)
        self.bscomp_input = QCheckBox()
        
        shapeLayout = QGridLayout()
        self.x_size_input = QLineEdit('256')
        self.y_size_input = QLineEdit('10')
        self.x_offset_input = QLineEdit('0')
        self.y_offset_input = QLineEdit('0')
        shapeLayout.addWidget(QLabel('Image Shape (x, y)'),0,0)
        shapeLayout.addWidget(self.x_size_input,0,1)
        shapeLayout.addWidget(self.y_size_input,0,2)
        shapeLayout.addWidget(QLabel('Offset (x, y)'),1,0)
        shapeLayout.addWidget(self.x_offset_input,1,1)
        shapeLayout.addWidget(self.y_offset_input,1,2)
        
        
        settingsLayout = QFormLayout()
        settingsLayout.addRow('Dwell Time (us)', self.dwell_input)
        settingsLayout.addRow('Metric', self.metric_input)
        settingsLayout.addRow('Initial Iterations', self.init_size_input)
        settingsLayout.addRow('Optimization Iterations', self.iterations_input)
        settingsLayout.addRow('Extra Iterations', self.extra_iterations_input)
        settingsLayout.addRow('Method', self.acq_func_input)
        settingsLayout.addRow('UCB Coefficient', self.ucb_coefficient_input)
        settingsLayout.addRow('Noise Level', self.noise_level_input)
        settingsLayout.addRow('Show Images', self.return_images_input)
        settingsLayout.addRow('Use Cross Correlation', self.ccorr_input)
        settingsLayout.addRow('Compensate with Beam Shift', self.bscomp_input)
        
        buttonsLayout = QGridLayout()
        
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_func)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_func)
        self.continue_button = QPushButton('Continue')
        self.continue_button.clicked.connect(self.continue_func)
        self.undo_button = QPushButton('Undo Last')
        self.undo_button.clicked.connect(self.undo_func)
        
        buttonsLayout.addWidget(self.start_button, 0, 0)
        buttonsLayout.addWidget(self.stop_button, 0, 1)
        buttonsLayout.addWidget(self.continue_button, 0, 2)
        buttonsLayout.addWidget(self.undo_button, 0, 3)
        
        controlPanelLayout.addLayout(abOptionsLayout)
        controlPanelLayout.addLayout(shapeLayout)
        controlPanelLayout.addLayout(settingsLayout)
        controlPanelLayout.addLayout(buttonsLayout)
        
        
        statusPanelLayout = QVBoxLayout()
        
        self.blank_surrogate = np.zeros((100,100))
        
        statusPanelLayout.addWidget(QLabel('Surrogate Model'))
        self.fig_surrogate, self.ax_surrogate = plt.subplots(1,1)      
        self.fig_surrogate.set_tight_layout(True)
        self.canvas_surrogate = FigureCanvas(self.fig_surrogate)
        statusPanelLayout.addWidget(self.canvas_surrogate)
        
        self.set_surrogate2D()
        
        x_toggle = QComboBox()
        x_toggle.addItems(['coarse', 'fine'])
        if self.ab_display_list[i] == 'C1' or self.ab_display_list[i] == 'C3':
            fc_toggle.addItems(['None'])
        else:
            fc_toggle.addItems(['coarse', 'fine'])
        self.fine_coarse.append(fc_toggle)
        
        
        statusPanelLayout.addWidget(QLabel('Status Box'))
        self.statusPanel = QTextEdit(readOnly=True)
        statusPanelLayout.addWidget(self.statusPanel)
        
        
        imagePanelLayout = QVBoxLayout()
        
        shape = (int(self.y_size_input.text()), int(self.x_size_input.text()))
        self.blank_image = np.zeros(shape)
        
        self.fig_before, self.ax_before = plt.subplots(1,1)
        self.ax_before.set_axis_off()
        self.ax_before.axis('equal') 
        self.imax_before = self.ax_before.matshow(self.blank_image)
        
        self.fig_before.set_tight_layout(True)
        self.canvas_before = FigureCanvas(self.fig_before)
        
        self.fig_after, self.ax_after = plt.subplots(1,1)
        self.ax_after.set_axis_off()
        self.ax_after.axis('equal')
        self.imax_after = self.ax_after.matshow(self.blank_image)
        
        self.fig_after.set_tight_layout(True)
        self.canvas_after = FigureCanvas(self.fig_after)
        
        imagePanelLayout.addWidget(QLabel('Initial Image'))
        imagePanelLayout.addWidget(self.canvas_before)
        imagePanelLayout.addWidget(QLabel('Final Image'))
        imagePanelLayout.addWidget(self.canvas_after)
        
        self.accept_button = QPushButton('Accept')
        self.accept_button.clicked.connect(self.accept_func)
        
        self.reject_button = QPushButton('Reject')
        self.reject_button.clicked.connect(self.reject_func)
        
        choiceLayout = QGridLayout()
        choiceLayout.addWidget(self.accept_button,0,0)
        choiceLayout.addWidget(self.reject_button,0,1)

        imagePanelLayout.addLayout(choiceLayout)
        
        outerLayout.addLayout(controlPanelLayout)
        outerLayout.addLayout(statusPanelLayout)
        outerLayout.addLayout(imagePanelLayout)
        
        self.setLayout(outerLayout)
        
        self.stop_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.accept_button.setEnabled(False)
        self.reject_button.setEnabled(False)
        
        print('GUI ready')
        
        self.ac_ae = BEACON_Client(host, port)
        
    def start_func(self):
        '''
        Function triggered by "start" button. Begins BEACON run with parameters in the GUI
        '''
        
        self.reset(buttons_reset=False)
        
        self.ac_ae.stop = False
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        ONE_BOX_CHECKED_FLAG = False # Check that at least one aberration was selected
        
        range_dict = {}
        
        for i in range(0,len(self.ab_list)):
            if self.check_boxes[i].isChecked():
                ONE_BOX_CHECKED_FLAG = True
                range_dict[self.ab_list[i]] = [int(self.lower_bounds[i].text()), 
                                               int(self.upper_bounds[i].text())]
                self.ac_ae.ab_select[self.ab_list[i]] = f'{self.fine_coarse[i].currentText()}'
                
        if not ONE_BOX_CHECKED_FLAG:
            self.msgPanel.append('Select at least one aberration')
            self.start_button.setEnabled(True)
        else:
            dwell_value = float(self.dwell_input.text())*1e-6
            shape_value = (int(self.x_size_input.text()), int(self.y_size_input.text()))
            offset_value = (int(self.x_offset_input.text()), int(self.y_offset_input.text()))
            
            self.blank_image = np.zeros(shape_value)
            self.imax_before.set_data(self.blank_image)
            self.imax_after.set_data(self.blank_image)
            
            init_size_value = int(self.init_size_input.text())
            iterations_value = int(self.iterations_input.text())
            
            acq_func_value = self.acq_func_input_names[self.acq_func_input.currentIndex()]
            metric_value = self.metric_input_names[self.metric_input.currentIndex()]
            ucb_coefficient_value = float(self.ucb_coefficient_input.text())
            
            noise_level_value = float(self.noise_level_input.text())
            
            return_images = self.return_images_input.isChecked()
            bscomp = self.bscomp_input.isChecked()
            ccorr = self.ccorr_input.isChecked()
            
            self.worker = Worker(self.ac_ae.ae_main,                                 
                                 range_dict,
                                 init_size_value, 
                                 iterations_value,
                                 acq_func_value,
                                 dwell_value, 
                                 shape_value,
                                 offset_value,
                                 metric_value,
                                 return_images,
                                 bscomp,
                                 ccorr,
                                 C1_defocus_flag=True,
                                 ab_select=None,
                                 return_dict=False,
                                 return_all_f_re=True,
                                 return_final_f_re=True,
                                 custom_early_stop_flag=False,
                                 ucb_coefficient=ucb_coefficient_value,
                                 noise_level=noise_level_value,
                                 init_hps=None,
                                 hp_bounds=None)
                                 
            self.thread_pool = QThreadPool()
            self.thread_pool.setMaxThreadCount(2)
            
            self.worker.signal.status.connect(self.on_status_data_changed)
            self.worker.signal.figure.connect(self.on_figure_data_changed)
            self.worker.signal.images.connect(self.on_images_data_changed)
            self.worker.signal.stopped.connect(self.on_stopped)
            self.thread_pool.start(self.worker)
    
    def stop_func(self):
        '''
        Function triggered by "stop" button. Stops run after it has been started.
        '''
        self.ac_ae.stop = True
        
    def accept_func(self):
        '''
        Function triggered by "accept" button. Accepts a suggested aberration change.
        '''
        self.ac_ae.accept_aberrations()
        self.undo_button.setEnabled(True)
        self.statusPanel.append('Corrections Accepted')
        self.reset()
        
    def reject_func(self):
        '''
        Function triggered by "accept" button. Rejects a suggested aberration change.
        '''
        self.statusPanel.append('Corrections Rejected')
        self.reset()
        
    def undo_func(self):
        '''
        Function triggered by "undo" button. Reverses the last accepted aberration change.
        '''
        self.ac_ae.undo_last()
        self.undo_button.setEnabled(False)
        
    def reset(self, buttons_reset=True):
        '''
        Resets GUI buttons after an aberration has been accepted or rejected
        '''
        if buttons_reset:
            self.accept_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            self.continue_button.setEnabled(False)
            self.start_button.setEnabled(True)
        
        shape_before = self.imax_before.get_array().shape
        shape = (int(self.y_size_input.text()), int(self.x_size_input.text()))
        self.blank_image = np.zeros(shape)
        
        if shape!=shape_before:
            self.ax_before.clear()
            self.ax_before.set_axis_off()
            self.ax_before.axis('equal') 
            
            self.ax_after.clear()
            self.ax_after.set_axis_off()
            self.ax_after.axis('equal')
        
        self.imax_before = self.ax_before.matshow(self.blank_image)
        self.canvas_before.draw()
        
        self.imax_after = self.ax_after.matshow(self.blank_image)
        self.canvas_after.draw()
        
        self.ax_surrogate.clear()
        self.set_surrogate2D()
        self.canvas_surrogate.draw()
        #if not self.ac_ae.SIM: self.ac_ae.ClientSocket.close() # Close once everything's done
    
    def continue_func(self):
        '''
        Function triggered by "continue" button. Continues the optimization run by 'extra_iterations'
        '''
        self.ac_ae.stop = False
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        extra_iterations_value = int(self.extra_iterations_input.text())
        
        self.worker = Worker(self.ac_ae.continue_training, extra_iterations=extra_iterations_value)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)
        
        self.worker.signal.status.connect(self.on_status_data_changed)
        self.worker.signal.figure.connect(self.on_figure_data_changed)
        self.worker.signal.images.connect(self.on_images_data_changed)
        self.worker.signal.stopped.connect(self.on_stopped)
        self.thread_pool.start(self.worker)
    
    @pyqtSlot(bytes) # connects to pyqtSignal object in receiver
    def on_status_data_changed(self, reply):
        '''
        Updates status panel.
        '''
        message = pickle.loads(reply)
        self.statusPanel.append(str(message))
    
    @pyqtSlot(bytes) # connects to pyqtSignal object in receiver
    def on_figure_data_changed(self, reply):
        '''
        Updates figure panel.
        '''
        f_re, x_data, y_data, error = pickle.loads(reply)
        if len(f_re.shape)==1:
            if not self.PLOT_IS_1D:
                self.set_surrogate1D()
            self.update_surrogate1D(f_re, x_data, y_data, error)
        elif len(f_re.shape)==2:
            if self.PLOT_IS_1D:
                self.set_surrogate2D()
            self.update_surrogate2D(f_re, x_data, y_data)
        
    @pyqtSlot(bytes) # connects to pyqtSignal object in receiver
    def on_images_data_changed(self, reply):
        '''
        Updates images panel.
        '''
        im_dict = pickle.loads(reply)
        image = im_dict['image']
        panel = im_dict['panel']
        if panel == 0:
            self.ax_before.axis('equal')
            self.imax_before.set_data(np.rot90(image))
            self.imax_before.set_clim(image.min(), image.max())
            self.canvas_before.draw()
        else:
            self.ax_after.axis('equal')
            self.imax_after.set_data(np.rot90(image))
            self.imax_after.set_clim(image.min(), image.max())
            self.canvas_after.draw()
    
    @pyqtSlot(int) # connects to pyqtSignal object in receiver
    def on_stopped(self, reply):
        '''
        Sets GUI after stop button pressed
        '''
        if reply == 1:
            self.stop_button.setEnabled(False)
            self.accept_button.setEnabled(True)
            self.reject_button.setEnabled(True)
            self.continue_button.setEnabled(True)
    
    def set_surrogate2D(self):
        #print('Setting 2D')
        self.ax_surrogate.set_axis_off()
        self.imax_surrogate = self.ax_surrogate.matshow(self.blank_surrogate)
        self.imax_surrogate_points = self.ax_surrogate.scatter([],[], s=200)
        self.canvas_surrogate.draw()
        self.PLOT_IS_1D = False
    
    def set_surrogate1D(self):
        #print('Setting 1D')     
        f_re = np.zeros(100)
        x = np.linspace(self.ac_ae.range_values[0][0], self.ac_ae.range_values[0][1], len(f_re))
        self.ax_surrogate.clear()
        self.ax_surrogate.set_xlim(self.ac_ae.range_values[0][0], self.ac_ae.range_values[0][1])
        self.ax_surrogate.set_ylim(np.min(f_re)-0.1, np.max(f_re)+0.1)
        self.ax_surrogate.axis('auto')
        self.imax_surrogate, = self.ax_surrogate.plot(x, f_re, lw=4, label=u'Prediction')
        self.imax_surrogate_points = self.ax_surrogate.scatter([],[], s=200, c='r', label=u'Observations')
        
        '''
        self.imax_surrogate_fill = self.ax_surrogate.fill(np.concatenate([x, x[::-1]]),
                                                          np.concatenate([x, x[::-1]]),
                                                          alpha=.5, fc='b', ec='None', label=u'Confidence Bound')
        '''
        
        self.canvas_surrogate.draw()
        self.PLOT_IS_1D = True
    
    def update_surrogate2D(self, f_re, x_data, y_data):
        #print('Updating 2D')
        self.imax_surrogate.set_data(f_re)
        self.imax_surrogate.set_clim(f_re.min(), f_re.max())
        x_data2 = np.interp(x_data, (-1,1), (0, 100))
        self.imax_surrogate_points.set_offsets(x_data2)
        self.imax_surrogate_points.set_array(y_data)
        self.imax_surrogate_points.set_cmap('magma')
        self.canvas_surrogate.draw()
    
    def update_surrogate1D(self, f_re, x_data, y_data, error):
        #print('Updating 1D')
        x = np.linspace(self.ac_ae.range_values[0][0], self.ac_ae.range_values[0][1], len(f_re))
        '''
        path = self.imax_surrogate_fill[0].get_paths()[0]
        path.vertices = np.column_stack([np.concatenate([x, x[::-1]]),
                                         np.concatenate([f_re-error,
                                                        (f_re+error)[::-1]])
                                         ])
        '''
        
        self.imax_surrogate.set_data(x, f_re)
        self.ax_surrogate.set_xlim(self.ac_ae.range_values[0][0], self.ac_ae.range_values[0][1])
        self.ax_surrogate.set_ylim(np.min((np.min(f_re), np.min(y_data)))-0.1, np.max((np.max(f_re), np.max(y_data)))+0.1)
        x_data2 = np.interp(x_data, (-1,1), (self.ac_ae.range_values[0][0], self.ac_ae.range_values[0][1]))
        self.imax_surrogate_points.set_offsets(np.hstack((x_data2, np.reshape(y_data, (len(y_data),1)))))
        self.canvas_surrogate.draw()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--serverhost', action='store', type=str, default='localhost', help='server host')
    parser.add_argument('--serverport', action='store', type=int, default=7001, help='server port')
    
    args = parser.parse_args()
    
    host = args.serverhost
    port = args.serverport
    
    app = QApplication(sys.argv)
    font = QFont('Sans Serif', 8)
    app.setFont(font, 'QLabel')
    app.setFont(font, 'QPushButton')
    app.setFont(font, 'QComboBox')
    app.setFont(font, 'QLineEdit')
    app.setFont(font, 'QCheckBox')
    app.setFont(font, 'QTextEdit')
    w = Widget(host, port)
    w.show()
    sys.exit(app.exec_())
