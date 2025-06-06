# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:14:54 2023

@author: alexa
"""

from gpcam.autonomous_experimenter import AutonomousExperimenterGP
import numpy as np
import sys
import pickle
import time
import zmq
import matplotlib.pyplot as plt
import argparse

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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


class BEACON_Client():
    def __init__(self, host, port, SIM=False, SOCKET_TEST=True, stop=False):
        
        self.stop = stop
        self.SIM = SIM
        self.last_saved_correction = {}
        self.image_list = []
        self.ab_select = {
                          'C1': None,
                          'A1_x': 'coarse',
                          'A1_y': 'coarse',
                          'B2_x': 'coarse',
                          'B2_y': 'coarse',
                          'A2_x': 'coarse',
                          'A2_y': 'coarse',
                          'C3': None,
                          'A3_x': 'coarse',
                          'A3_y': 'coarse',
                          'S3_x': 'coarse',
                          'S3_y': 'coarse',
                          }
        
        try:
            context = zmq.Context()
            self.ClientSocket = context.socket(zmq.REQ)
            self.ClientSocket.connect(f"tcp://{host}:{port}")
            print(f'Connected to BEACON server at {host}:{port}')
        except ConnectionRefusedError:
            print('Start the BEACON server')
            exit()
        
        if SOCKET_TEST:
            d = {'type': 'ping'}
            Response = self.send_traffic(d)
            qval = Response['reply_data']
            print(qval)
        
        self.status_callback = None
        self.figure_callback = None
        self.images_callback = None
        self.stopped_callback = None
        
        self.noise_value = None
        
        self.init_points = None
    
    def send_traffic(self, message):
        '''
        Sends and receives messages from the server.
        
        Parameters
        ----------
        message : dict
            Message for the server.
        
        Returns
        -------
        response : dict
            Response from the server.
        '''
        
        self.ClientSocket.send(pickle.dumps(message))
        response = pickle.loads(self.ClientSocket.recv())
        return response
    
    def set_ref(self, dwell, shape, offset=(0,0), return_images=True):
        '''
        Sets the reference image.
        
        Parameters
        ----------
        dwell : float
            Dwell time in seconds
        shape : tuple or array
            Image shape in pixels
        return_images : bool
            Flag to return image to the GUI display
        '''
        d = {'type': 'ref',
             'dwell': dwell,
             'shape': shape,
             'offset': offset,
             'return_images': return_images,
             }
        Response = self.send_traffic(d)
        if self.status_callback is not None:
            self.status_callback.emit(pickle.dumps('Reference image set'))
        im_dict = {'image': Response['reply_data'],
                   'panel': 0}
        if return_images and self.images_callback is not None:
            self.images_callback.emit(pickle.dumps(im_dict))
    
    def get_image(self, ab_values={}):
        '''
        Acquire image with specified aberrations.
        Aberrations are reset to current values after image is taken.
        
        Parameters
        ----------
        ab_values : dict
            Dictionary of aberration names and magnitudes that need to be changed.
        '''
        #print(ab_values)
        for k, v in ab_values.items():
            if v > 1e-4:
                print(f'{k} ab_value is > 1e-4')
        
        d = {'type': 'ac',
             'ab_values': ab_values,
             'ab_select': self.ab_select,
             'dwell': self.dwell,
             'shape': self.shape,
             'offset': self.offset,
             'metric': self.metric,
             'C1_defocus_flag': self.C1_defocus_flag,
             'return_images': self.return_images or self.return_dict,
             'bscomp': self.bscomp,
             'ccorr': self.ccorr,
             }
        return self.send_traffic(d)
    
    def ab_only(self, ab_values, C1_defocus_flag=True, undo=False, bscomp=False):
        '''
        Change aberrations without acquiring image.
        Aberrations are NOT reset to current values after function call.
        
        Parameters
        ----------
        ab_values : dict
            Dictionary of aberration names and magnitudes that need to be changed.
        C1_defocus_flag : bool
            True: Use microscope defocus to correct C1.
            False: Use aberration corrector to correct C1.
        undo : bool
            Apply the negative of ab_vals to change the aberrations
        bscomp : bool
            Use beam shift to compensate for changes in field of view when changing aberrations.
        '''
        d = {'type': 'ab_only',
             'ab_values': ab_values,
             'ab_select': self.ab_select,
             'C1_defocus_flag': C1_defocus_flag,
             'undo': undo,
             'bscomp': bscomp,
             }
        Response = self.send_traffic(d)
        print(Response)
    
    def normalization(self):
        '''
        Acquire images for normalization calculations
        '''
        self.norm_points = [{},{},{}]
        
        for i in range(len(self.ab_keys)):
            self.norm_points[0][f'{self.ab_keys[i]}'] = self.ranges[f'{self.ab_keys[i]}'][0]*1e-9
            self.norm_points[1][f'{self.ab_keys[i]}'] = 0*1e-9
            self.norm_points[2][f'{self.ab_keys[i]}'] = self.ranges[f'{self.ab_keys[i]}'][1]*1e-9
        
        self.norm_values = []
        self.norm_image_dict = []
        
        for i in range(3):
            ab_values = self.norm_points[i]
            Response = self.get_image(ab_values)
            n_values, nim_dict = Response['reply_data']
            self.norm_values.append(n_values)
            self.norm_image_dict.append(nim_dict)

    def initial_points(self, n):
        '''
        Acquire initial images
        '''
        self.init_points = []
        
        for i in range(n):
            init_point_dict = {}
            for j in range(len(self.ab_keys)):
                init_point_dict[f'{self.ab_keys[j]}'] = np.random.uniform(self.ranges[f'{self.ab_keys[j]}'][0]*1e-9, self.ranges[f'{self.ab_keys[j]}'][1]*1e-9)
            self.init_points.append(init_point_dict)
        
        self.init_values = []
        self.init_image_dict = []
        
        for i in range(n):
            ab_values = self.init_points[i]
            Response = self.get_image(ab_values)
            i_values, iim_dict = Response['reply_data']
            self.init_values.append(i_values)
            self.init_image_dict.append(iim_dict)
            
    def custom_ucb_func(self, x, obj):
        '''
        Custom acquisition function.
        
        Parameters
        ----------
        x : array
            Array of measurement locations
        obj : gp_optimizer class
            Instance of gpcam.AutonomousExperimenterGP.gp_optimizer
        '''
        
        mean = obj.posterior_mean(x)["f(x)"]
        cov = obj.posterior_covariance(x)["v(x)"]
        return mean + self.ucb_coefficient * np.sqrt(cov)

    def custom_noise(self, x, hps, obj):
        '''
        Custom noise calculation function.
        
        Parameters
        ----------
        x : array
            Array of measurement locations
        hps: array
            Numpy array containing gpcam hyperparamaters
        obj : gp_optimizer class
            Instance of gpcam.AutonomousExperimenterGP.gp_optimizer
        '''
        
        #print(self.noise_level)
        self.noise_value = self.noise_level/self.norm_range#*abs(np.mean(obj.y_data))
        #print(self.noise_value)
        #print(abs(np.mean(obj.y_data))*self.noise_level/self.norm_range)
        #self.noise_value = abs(np.mean(obj.y_data))*self.noise_level
        return np.identity(len(obj.y_data))*self.noise_value

    def instrument(self, data):
        '''
        Instrument function for gpcam.
        
        Parameters
        ----------
        data : list
            Data from gpcam
        
        Returns
        -------
        data : list
            Updated data from gpcam
        '''
        
        for entry in data:
            # breakpoint()
            ab_values = {}
            for i in range(len(self.ranges)):
                ab_values[f'{self.ab_keys[i]}'] = np.interp(entry["x_data"][i], (-1, 1), (self.range_values[i][0], self.range_values[i][1]))*1e-9
            
            Response = self.get_image(ab_values)
            
            if self.return_images or self.return_dict:
                ret_value, im_dict = Response['reply_data']
                if self.images_callback is not None:
                    im_dict['panel'] = 1
                    self.images_callback.emit(pickle.dumps(im_dict))
                if self.return_dict:
                    self.image_list.append(im_dict['image'])
            else:
                ret_value = Response['reply_data']
            
            entry["y_data"] = (ret_value-self.norm_min)/self.norm_range

        return data

    def run_in_every_iter(self, obj):
        '''
        Function to run in every iteration
        
        Parameters
        ----------
        obj : gp_optimizer class
            Instance of gpcam.AutonomousExperimenterGP.gp_optimizer
        '''
        x_data = obj.x_data[-1]
        ndims = len(x_data)
        x_data_interp = np.zeros(ndims)
        for i in range(ndims):
            x_data_interp[i] = np.interp(x_data[i], (-1, 1), (self.range_values[i][0], self.range_values[i][1]))
        y_data = obj.y_data[-1]
        
        # breakpoint()
        print(len(obj.x_data), np.array2string(x_data_interp, precision=2, floatmode='fixed'), '{:.2f}'.format(y_data))

        # Update GUI status bar
        if self.status_callback is not None:
            status_reply = pickle.dumps(f'{len(obj.x_data)}, {np.array2string(x_data_interp, precision=2, floatmode="fixed")}, {"{:.2f}".format(y_data)}')
            self.status_callback.emit(status_reply)
        
        # Append this iteration's hyperparameters to the list
        if self.return_dict:
            self.hps_list.append(obj.gp_optimizer.hyperparameters)
        
        # Return the shape of the surrogate model
        if self.return_all_f_re:
            if ndims==2 or ndims==3:
                f_re = self.get_f_re(obj)
                self.f_re_list.append(f_re)
                if self.figure_callback is not None:
                    figure_reply = pickle.dumps((f_re, obj.x_data, obj.y_data, None))
                    self.figure_callback.emit(figure_reply)
            elif ndims==1:
                f_re, v_re = self.get_f_re(obj, get_v_re=True)
                self.f_re_list.append(f_re)
                if self.figure_callback is not None:
                    figure_reply = pickle.dumps((f_re, obj.x_data, obj.y_data, v_re*self.ucb_coefficient))
                    self.figure_callback.emit(figure_reply)
            '''
            elif ndims==3 or ndims==4:
                f_re = self.get_f_re(obj, points=50)
                self.f_re_list.append(f_re)
            '''
        ACTIVE = True
        if self.return_model_max_list:
            if ndims==7 and len(obj.x_data) not in np.arange(20)*10 and ACTIVE:
                self.model_max_list.append(None) 
            else:
                model_max = obj.gp_optimizer.ask(bounds=self.parameters, acquisition_function='maximum')['x'][0]
                for i in range(len(self.ranges)):
                    model_max[i] = np.interp(model_max[i], (-1, 1), (self.range_values[i][0], self.range_values[i][1]))
                self.model_max_list.append(model_max)
    
    def get_f_re(self, obj, points=50, get_v_re=False):
        '''
        Get shape of the surrogate model
        
        Parameters
        ----------
        obj : gp_optimizer class
            Instance of gpcam.AutonomousExperimenterGP.gp_optimizer
        points : int
            Resolution with which to sample the surrogate model
        
        Returns
        -------
        f_re : array
            Numpy array containing shape of the surrogate model
        '''
        ndims = len(obj.x_data[-1])
        dims = [None]*ndims
        for i in range(ndims):
            dims[i] = np.linspace(self.parameters[0][0], self.parameters[0][1], points)
        mdims = np.meshgrid(*dims)
        mdims_flat = [None]*ndims
        for i in range(ndims):
            mdims_flat[i] = mdims[i].ravel()
        x_pred = np.stack(mdims_flat).T
        
        shape = tuple([points]*ndims)
        f = obj.gp_optimizer.posterior_mean(x_pred)["f(x)"]
        f_re = np.reshape(f,shape)
        
        if get_v_re:
            v = obj.gp_optimizer.posterior_covariance(x_pred)["v(x)"]
            v_re = np.reshape(v,shape)
            return f_re, v_re
        else:
            return f_re
    
    def create_dict(self):
        '''
        Creates a dictionary containing all the data and metadata from a single run.
            
        Returns
        -------
        self.BEACON_dict : dict
            Dictionary of data and metadata
        '''
        self.BEACON_dict = {'x': self.ae.x_data, # NEEDS TO BE RESCALED
                           'y': self.ae.y_data,
                           'range_dict': self.ranges,
                           'init_size': self.init_size,
                           'func': self.acq_func,
                           'dwell': self.dwell,
                           'shape': self.shape,
                           'offset': self.offset,
                           'metric': self.metric,
                           'bscomp': self.bscomp,
                           'ccorr': self.ccorr,
                           'norm_min': self.norm_min,
                           'norm_range': self.norm_range,
                           'noise_level': self.noise_value,
                           'initial_image': self.initial_image,
                           'final_image': self.final_image,
                           'image_list': self.image_list,
                           'f_re_list': self.f_re_list,
                           'hps_list': self.hps_list,
                           'final_f_re': self.final_f_re,
                           'model_max': self.model_max,
                           'model_max_list': self.model_max_list,
                           }
        
        if self.ucb_coefficient is not None:
            self.BEACON_dict['func'] = f'ucb-{self.ucb_coefficient}'
        
        return self.BEACON_dict

    def ae_main(self,
                ranges,
                init_size,
                iterations,
                func,
                dwell, 
                shape,
                offset,
                metric,
                return_images,
                bscomp,
                ccorr,
                C1_defocus_flag=True,
                include_norm_runs=False,
                ab_select=None,
                return_dict=False,
                return_all_f_re=False,
                return_final_f_re=False,
                return_model_max_list=False,
                custom_early_stop_flag=False,
                ucb_coefficient=2,
                noise_level=0.1,
                init_hps=None,
                hp_bounds=None,
                status_callback=None,
                figure_callback=None,
                images_callback=None,
                stopped_callback=None):
        '''
        Main function for setting up and starting BEACON run
        
        Parameters
        ----------
        ranges : list
            Search ranges for each of the parameters
        init_size : int
            Number of initial random searches before BEACON begins
        iterations : int
            Initial number of iterations for a single run [CONFUSING NAMING]
        func : 
            Name of the function to be used [WHAT OPTIONS?]
        dwell : float
            Dwell time in seconds
        shape : tuple or array
            Image shape in pixels
        offset : tuple or array
            Offset in pixels
        metric : str
            Name of the metric to be used (normalized variance, variance, standard deviation)
        return_images : bool
            Flag to return image to the GUI display
        bscomp : bool
            Use beam shift to compensate for changes in field of view when changing aberrations (NOT RECOMMENDED)
        ccorr : bool
            Use cross-correlation to compensate for changes in field of view when changing aberrations (STRONGLY RECOMMENDED)
        C1_defocus_flag : bool
            True: Use microscope defocus to correct C1.
            False: Use aberration corrector to correct C1.
        include_norm_runs : bool
            Flag to include normalization runs in initial dataset
        ab_select : dict
            Dictionary of aberration names and whether to use coarse or fine correction.
        return_dict : bool
            Choose to return the dictionary of data and metadata
        return_all_f_re : bool
            Calculate and save surrogate models for every iteration (slow in higher dimensions)
        return_final_f_re : bool
            Calculate and save the final surrogate model
        return_model_max_list : bool
            Calculate and save the model maximum after every iteration
        custom_early_stop_flag : bool
            Use early stopping criterion (NOT CURRENTLY IMPLEMENTED)
        ucb_coefficient : float
            UCB factor used in the custom_ucb_func (increasing favors exploration, decreasing favors exploitation)
        noise_level : float
            Noise level for custom_noise function
        init_hps : list
            List of initial hyperparameters
        hp_bounds : list
            List of hyperparamater bounds
        status_callback : pyqt worker signal
            pyqt worker signal for GUI status bar
        figure_callback : pyqt worker signal
            pyqt worker signal for GUI figure (surrogate model)
        images_callback : pyqt worker signal
            pyqt worker signal for GUI images plotting
        stopped_callback : pyqt worker signal
            pyqt worker signal for stopping BEACON run
        '''
        
        self.status_callback = status_callback
        self.figure_callback = figure_callback
        self.images_callback = images_callback
        self.stopped_callback = stopped_callback
        
        self.return_dict = return_dict
        self.return_all_f_re = return_all_f_re
        self.return_final_f_re = return_final_f_re
        self.return_model_max_list = return_model_max_list

        self.image_list = []
        self.f_re_list = []
        self.model_max_list = []
        self.hps_list = []
        self.final_f_re = None

        self.ranges = ranges
        self.init_size = init_size
        self.ucb_coefficient = ucb_coefficient
        self.noise_level = noise_level
        
        if self.ucb_coefficient is None:
            self.ucb_coefficient = 3
            
        self.acq_func = self.custom_ucb_func
        
        self.dwell = dwell
        self.shape = shape
        self.offset = offset
        self.metric = metric
        self.bscomp = bscomp
        self.ccorr = ccorr
        
        self.return_images = return_images
        if self.return_images:
            if self.ccorr:
                self.image_stack = np.zeros((0,int(self.shape[0]/2),int(self.shape[1]/2))) # normally /2
            else:
                self.image_stack = np.zeros((0,int(self.shape[0]),int(self.shape[1])))
            self.resolutions = []
        self.C1_defocus_flag = C1_defocus_flag
        
        self.range_values = list(ranges.values())
        self.ab_keys = list(ranges.keys())
        #self.parameters = np.array(list(ranges.values()))
        
        ndims = len(self.ranges)

        self.parameters = np.repeat(np.array([[-1,1]]), ndims, axis=0)
        
        if init_hps is None:
            self.init_hps = np.ones(ndims+1)
        else:
            self.init_hps = init_hps
            
        if hp_bounds is None:
            hp_bounds_0 = np.array([[1e-2,2e0]])
            hp_bounds_1 = np.repeat(np.array([[1e-2,1e0]]), ndims, axis=0)
            self.hp_bounds = np.vstack((hp_bounds_0, hp_bounds_1))
        else:
            self.hp_bounds = hp_bounds
        
        if len(self.init_hps)!=len(self.hp_bounds): raise ValueError('init_hps and hp_bounds have different sizes')
        
        if self.status_callback is not None:
            self.status_callback.emit(pickle.dumps('Normalizing'))
        
        self.set_ref(dwell, shape)
        
        self.normalization()
        self.norm_min = np.min(self.norm_values)
        self.norm_range = np.ptp(self.norm_values)
        
        time.sleep(2)
        
        #self.norm_min = 0 # FOR TESTING ONLY
        #self.norm_range = 1 # FOR TESTING ONLY
        
        if include_norm_runs:
            n = np.max((0, self.init_size-3))
            self.initial_points(n)
            self.init_points = self.init_points + self.norm_points
            self.init_values = self.init_values + self.norm_values
        else:
            self.initial_points(self.init_size)
            
        self.init_values = list((self.init_values-self.norm_min)/self.norm_range)
            
        self.init_ab_values = np.zeros((self.init_size, ndims))
        self.init_ab_values_scaled = np.zeros((self.init_size, ndims))
        
        for i in range(self.init_size):
            self.init_ab_values[i] = list(self.init_points[i].values())
            for j in range(ndims):
                self.init_ab_values_scaled[i][j] = np.interp(self.init_ab_values[i][j]*1e9, (self.range_values[j][0], self.range_values[j][1]), (-1,1))
        
        self.ae = AutonomousExperimenterGP(self.parameters,
                                           self.init_hps,
                                           self.hp_bounds,
                                           instrument_function=self.instrument,
                                           x_data=self.init_ab_values_scaled,
                                           y_data=self.init_values,
                                           acquisition_function=self.acq_func,
                                           noise_function=self.custom_noise,
                                           run_every_iteration=self.run_in_every_iter,
                                           compute_device='cpu',
                                           )
        
        for i in range(self.init_size):
            x_data_interp = np.zeros(ndims)
            for j in range(ndims):
                x_data_interp[j] = np.interp(self.ae.x_data[i][j], (-1, 1), (self.range_values[j][0], self.range_values[j][1]))
            print(i+1, np.array2string(x_data_interp, precision=2, floatmode='fixed'), '{:.2f}'.format(self.ae.y_data[i]))
            # Update GUI status bar
            if self.status_callback is not None:
                status_reply = pickle.dumps(f'{i+1}, {np.array2string(x_data_interp, precision=2, floatmode="fixed")}, {"{:.2f}".format(self.ae.y_data[i])}')
                self.status_callback.emit(status_reply)
        
        self.ae_run(iterations)
        
    def ae_run(self, iterations, retraining_list=None):
        '''
        Run BEACON for a given number of iterations
        
        Parameters
        ----------
        iterations : int
            Number of BEACON iterations
        retraining_list : list
            Iterations at which to retrain the hyperparameters
        '''
        if retraining_list == None:
            retraining_list = list(np.arange(0,iterations,10))
        
        N = len(self.ae.x_data)
            
        for i in range(iterations):
            if not self.stop:
                N+=1
                # try statement here
                self.ae.go(N = N, retrain_globally_at=retraining_list, # retraining list in function declaration
                           acq_func_opt_setting = lambda number: "global",
                           #custom_early_stop_flag=custom_early_stop_flag # Resurrect this
                           )
            else:
                break
            
        self.model_max = self.ae.gp_optimizer.ask(bounds=self.parameters, acquisition_function='maximum')['x'][0]
        self.model_max_val = self.ae.gp_optimizer.ask(bounds=self.parameters, acquisition_function='maximum')['f(x)'][0]
        mm_ab_keys = list(self.ranges.keys())
        self.mm_ab_values = {}
        
        for i in range(len(self.ranges)):
            self.model_max[i] = np.interp(self.model_max[i], (-1, 1), (self.range_values[i][0], self.range_values[i][1]))
            self.mm_ab_values[mm_ab_keys[i]] = self.model_max[i]*1e-9
        
        mmstr = np.array2string(self.model_max, precision=2, floatmode='fixed')
        print("model max =", mmstr, self.model_max_val)
        
        #print(self.ae.x_data, self.ae.y_data)
        #print(mm_ab_values)
               
        self.ccorr = False
        
        Response = self.get_image({})
        _, self.initial_image = Response['reply_data']
        
        Response = self.get_image(self.mm_ab_values)
        _, self.final_image = Response['reply_data']
        
        if self.return_images and self.images_callback is not None:
            self.initial_image['panel'] = 0
            self.images_callback.emit(pickle.dumps(self.initial_image))
            self.final_image['panel'] = 1
            self.images_callback.emit(pickle.dumps(self.final_image))
        
        if self.status_callback is not None:
            self.status_callback.emit(pickle.dumps(f'model max = {mmstr}'))
            self.status_callback.emit(pickle.dumps('Done'))
        
        if self.stopped_callback is not None:
            self.stopped_callback.emit(1)
        
        if self.return_final_f_re:
            self.final_f_re = self.get_f_re(self.ae)
        
        if self.return_dict:
            self.create_dict()
        
        print('Done')
    
    def continue_training(self, extra_iterations=5,
                          status_callback=None,
                          figure_callback=None,
                          images_callback=None,
                          stopped_callback=None):
        '''
        Continue BEACON run for a given number of iterations
        
        Parameters
        ----------
        extra_iterations : int
            Number of BEACON iterations by which to continue the run
        status_callback : pyqt worker signal
            pyqt worker signal for GUI status bar
        figure_callback : pyqt worker signal
            pyqt worker signal for GUI figure
        images_callback : pyqt worker signal
            pyqt worker signal for GUI images plotting
        stopped_callback : pyqt worker signal
            pyqt worker signal for stopping BEACON run
        '''
        
        self.stop = False
        
        self.status_callback = status_callback
        self.figure_callback = figure_callback
        self.images_callback = images_callback
        self.stopped_callback = stopped_callback
        
        print('Continue Function Called')
        
        if self.ae is not None:
            self.ae_run(extra_iterations)
        else:
            print('self.ae does not exist')
    
    def accept_aberrations(self):
        self.ab_only(self.mm_ab_values,
                     C1_defocus_flag=self.C1_defocus_flag,
                     bscomp=self.bscomp)
        
        self.last_saved_correction = self.mm_ab_values
        
    def undo_last(self):
        undo_ab_values = self.mm_ab_values.copy()
        for name in list(undo_ab_values.keys()):
            undo_ab_values[name] = -undo_ab_values[name]
        print(undo_ab_values)
        self.ab_only(undo_ab_values, 
                     C1_defocus_flag=self.C1_defocus_flag, 
                     undo=True, bscomp=self.bscomp)
        
        self.last_saved_correction = {}
    """
    # UNUSED
    def rebuild(self, hps=None, x_data=None, y_data=None):
        
        '''
        Rebuild instance of gpcam.AutonomousExperimenterGP with given data and hyperparameters
        
        Parameters
        ----------
        hps : list
            Hyperparameters
        x_data : list
            Sampled points in aberration space
        y_data : list
            Metric values corresponding to x_data
        '''
        
        if hps is None:
            hps = self.ae.gp_optimizer.hyperparameters
        if x_data is None:
            x_data = self.ae.x_data
        if y_data is None:
            y_data = self.ae.y_data
            
        self.ae_2 = AutonomousExperimenterGP(self.parameters, hps,
                                             self.hyperparameter_bounds,
                                             x_data=x_data,
                                             y_data=y_data, 
                                             acq_func=self.custom_ucb_func,
                                             compute_device="cpu")
        
        #self.ae_2.train()
        
        self.final_f_re_2 = self.get_f_re(self.ae_2)
        self.model_max_2 = self.ae_2.gp_optimizer.ask(bounds=self.parameters, acquisition_function='maximum')['x'][0]
    """
    
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

'''
To-do
Plot points on scatter - Unimportant
'''