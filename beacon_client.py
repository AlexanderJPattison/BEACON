from gpcam.autonomous_experimenter import AutonomousExperimenterGP
import numpy as np
import pickle
import time
import zmq

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
    
