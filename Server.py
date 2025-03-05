# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:14:26 2023

@author: alexa
"""

import zmq
import numpy as np
import pickle
import argparse
import socket
import json
import pynetstring

# For connections to FEI TEMScripting and TIA
from comtypes.client import CreateObject
from comtypes.safearray import safearray_as_ndarray

class CorrectorCommands():
    '''
    Adapted from CorrectorServer.py provided by CEOS, GmbH.
    '''
    def __init__(self, host='localhost', port=7072, verbose=False):
        self.host = host
        self.port = port
        self.v = verbose
        
    def communicate(self, name, parameter=None):
        '''
        Send JSON string to aberration corrector
        
        Parameters
        ----------
        name : str
            Name of command
        parameter : str
            A dict or list of parameters
        '''
        data = self.encodeJSON(name, parameter)
        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            # Connect to server and send data
            sock.connect((self.host, self.port))
            sock.sendall(pynetstring.encode(data))
                
            # Receive data from the server and shut down
            received = sock.recv(1024)
            received = pynetstring.decode(received)
        
        finally:
            sock.close()
        
        if self.v:
            print('Sent:     {}'.format(data))
            print('Received: {}'.format(received))
        
        return received
    
    def encodeJSON(self, name, parameter=None):
        '''
        Send a RPC request to the server.
    
        Parameters
        ----------
        name : str
            Name of command
        parameter : str
            A dict or list of parameters
        '''
        if self.v:
            print(name)
            print(parameter)
        
        if parameter is None:
            parameter = {}
        
        JSON_dict = {'jsonrpc': '2.0',
                     'id': 1,
                     'method': name,
                     'params': parameter}
                     
        return json.dumps(JSON_dict)
        
    def correctAberration(self, name='A1', value=[0,0], target=[0,0], select=None):
        '''
        Correct the aberration currently selected in GUI. Either use value
        from last measurement stored in server or entered value.
        
        Parameters
        ----------
        name : str
            Name of command. Choices are:
            'C1', 'A1', 'A2', 'B2', 'C3', 'A3', 'S3', 'A4', 'D4', 'B4', 'C5', 'A5', 'R5', 'S5', 'We', 'WD'
        value : list
            x and y values by which to offset aberration from current state. Unit is m
        target : list
            Target x and y values for aberrations. Unit is m. NOT USED IN THIS PROGRAM
        select :
            Select coils by which to change aberration. Choices are:
            '', 'coarse', 'fine', 'condenser', 'projector', 'objective'
        
        Returns
        -------
        A Deferred that fires when the command has finished
        
        '''
        params = {'name': name,
                  'value': value,
                  'target': target,
                  'select': select
                  }
                  
        return self.communicate('correctAberration', params)
        
    def getInfo(self):
        '''
        Fetch various information from the corrector software.
        
        Returns
        -------
        A dict containing various information
        '''
        return self.communicate('getInfo')
    
    def measureC1A1(self):
        """
        Do a single C1A1(B2A2WD) measurement.
        
        :returns: a Deferred containing the aberrations as dict
        """
        return self.communicate('measureC1A1')
    
class TIA_control():
    def __init__(self):
        # Connect to the microscope
        self._microscope = CreateObject('TEMScripting.Instrument')
        self.TIA = CreateObject('ESVision.Application')
        self.Acq = self._microscope.Acquisition
        self.Ill = self._microscope.Illumination
        self.Proj = self._microscope.Projection
        self.Stage = self._microscope.Stage
        
        # Connect to STEM
        detector0 = self.Acq.Detectors(0)
        # Add the first detector
        self.Acq.AddAcqDevice(detector0)
        
        self.TIA.ScanningServer().AcquireMode = 1 #0=continuous, 1=single
        self.TIA.ScanningServer().ScanMode = 2 #0=spot, 1=line, 2=frame

    def open_column_valve(self):
        self._microscope.Vacuum.ColumnValvesOpen = True
        print('Column valves open')

    def close_column_valve(self):
        self._microscope.Vacuum.ColumnValvesOpen = False
        print('Column valves closed')

    def create_or_set_display_window(self, sizeX, sizeY):
        self.window_name = 'BEACON image'
        winlist = self.TIA.DisplayWindowNames()
        found = False
        for ii in range(winlist.count):
            if winlist[ii] == self.window_name:
                found = True
                
        if found:
            self.w2D = self.TIA.FindDisplayWindow(self.window_name)
            self.d1 = self.w2D.FindDisplay('Image 1 Display')
            if self.d1 is not None:
                self.disp = self.d1.Image
            else:
                self.d1 = self.w2D.addDisplay('Image 1 Display', 0,0,3,1)
                self.disp = self.d1.AddImage('Image 1', sizeX, sizeY, self.TIA.Calibration2D(0,0,1,1,0,0))
        else:
            self.w2D = self.TIA.AddDisplayWindow()
            self.w2D.name = self.window_name
            self.d1 = self.w2D.addDisplay('Image 1 Display', 0,0,3,1)
            self.disp = self.d1.AddImage('Image 1', sizeX, sizeY, self.TIA.Calibration2D(0,0,1,1,0,0))

    def get_mag(self):
        return self.Ill.StemMagnification

    def set_mag(self, mag):
        self.Ill.StemMagnification = mag
        print('Mag set to {}'.format(self.Ill.StemMagnification))
        
    def get_stage_pos(self):
        ''' Get stage position '''
        stageObj = self.Stage.Position
        print('Stage position = {}'.format(stageObj))
        return stageObj.X, stageObj.Y, stageObj.Z, stageObj.A, stageObj.B

    def move_stage_delta(self, dX=0, dY=0, dZ=0, dA=0, dB=0):
        ''' Move stage by delta value '''
        #n = int('{}{}{}{}{}'.format(int(dB==1), int(dA==1), int(dZ==1), int(dY==1), int(dX==1)), 2)
        n = 15
        print('Moving by {}, {}, {}, {}, {}'.format(dX, dY, dZ, dA, dB))
        stageObj = self.Stage.Position
        stageObj.X += dX
        stageObj.Y += dY
        stageObj.Z += dZ
        stageObj.A += dA
        stageObj.B += dB
        self.Stage.GoTo(stageObj, n)
        #print('Stage moved to = {}'.format(self.Stage.Position()))

    def blank(self):
        ''' Blanks beam '''
        self.Ill.BeamBlanked = True
        print('Beam blanked')
    
    def unblank(self):
        ''' Unblanks beam '''
        self.Ill.BeamBlanked = False
        print('Beam unblanked')
    
    def change_defocus(self, df):
        '''
        Changes the defocus
        
        Parameters
        ----------
        df : float
            Amount of defocus to change (in metres)
        '''
        print('Changing defocus by {}'.format(df))
        currentDF = self.Proj.Defocus
        self.Proj.Defocus = currentDF + df
        print('Defocus set to {}'.format(self.Proj.Defocus))
        
    def microscope_acquire_image(self, dwell, shape, offset=(0,0)):
        '''
        Acquire image in TIA
        
        Parameters
        ----------
        dwell : float
            Dwell time
        shape : tuple, array
            Image shape
        offset : typle, array
            Offset of image from current center (might be issues if more than one value is non-zero)
        
        Returns
        -------
        image_data : array
            Acquired image
        '''
        
        sizeX = shape[0]
        sizeY = shape[1]
        centerX = offset[0]
        centerY = offset[1]
        
        print('Acquiring image with shape = {}, {}, offset = {}, {}'.format(sizeX, sizeY, centerX, centerY))
        
        if self.TIA.AcquisitionManager().IsAcquiring:
            self.TIA.AcquisitionManager().Stop()
        
        self.create_or_set_display_window(sizeX, sizeY)

        scrange = self.TIA.ScanningServer().GetTotalScanRange

        length = np.maximum(sizeX, sizeY)
        startX = scrange.StartX/length*sizeX
        endX = scrange.EndX/length*sizeX
        startY = scrange.StartY/length*sizeY
        endY = scrange.EndY/length*sizeY
        resolution = (endX-startX)/sizeX
        
        self.TIA.ScanningServer().SetFrameScan(self.TIA.Range2D(startX,startY,endX,endY), resolution) # can resolution be different in x and y?
        self.TIA.ScanningServer().DwellTime = dwell
        
        calX = self.TIA.ScanningServer().ScanResolution
        calY = self.TIA.ScanningServer().ScanResolution
        
        # Needed in case someone runs search between BEACON searches
        self.TIA.ScanningServer().AcquireMode = 1 #0=continuous, 1=single
        self.TIA.ScanningServer().ScanMode = 2 #0=spot, 1=line, 2=frame
        
        self.TIA.AcquisitionManager().LinkSignal('Analog3', self.d1.Image)
        
        self.unblank()
        self.TIA.AcquisitionManager().Start()
        while self.TIA.AcquisitionManager().IsAcquiring:
            pass
        self.blank()

        data = self.disp.Data
        image_data = np.array(data.Array)
        unit1 = self.d1.SpatialUnit # returns SpatialUnit object
        unitName = unit1.unitstring # returns a string (such as nm)

        return image_data, calX, calY, unitName

    def microscope_acquire_image_old(self, dwell, shape, offset=(0,0)):
        '''
        Acquire image in TIA
        Todo: Remove this once you figure out the TIA window issue
        
        Parameters
        ----------
        dwell : float
            Dwell time
        shape : tuple, array
            Image shape
        offset : typle, array
            Offset of image from current center (might be issues if more than one value is non-zero)
        
        Returns
        -------
        image_data : array
            Acquired image
        '''
        if shape[0] > 512:
            binning = 4
        else:
            binning = 8
        
        myStemSearchParams = self.Acq.Detectors.AcqParams
        myStemSearchParams.Binning = binning
        myStemSearchParams.ImageSize = 0 # Size of image (0 = full size, 1 = half size, 2 = quarter size)
        myStemSearchParams.DwellTime = dwell
        self.Acq.Detectors.AcqParams = myStemSearchParams
        
        if self.TIA.AcquisitionManager().isAcquiring:
            self.TIA.AcquisitionManager().Stop()
        self.unblank()
        # Acquire an image
        acquiredImageSet = self.Acq.AcquireImages()
        with safearray_as_ndarray:
            image_data = acquiredImageSet(0).AsSafeArray # get data as ndarray
        self.blank()
        
        window1 = self.TIA.ActiveDisplayWindow()
        Im1 = window1.FindDisplay(window1.DisplayNames[0]) #returns an image display object
        unit1 = Im1.SpatialUnit #returns SpatialUnit object
        unitName = unit1.unitstring #returns a string (such as nm)
        calX = self.TIA.ScanningServer().ScanResolution
        calY = self.TIA.ScanningServer().ScanResolution

        return image_data, calX, calY, unitName

class CEOS_RPC_control():
    def __init__(self, rpchost, rpcport):
        print('Attempting to connecting to CEOS RPC gateway at '+str(rpchost)+':'+str(rpcport))
        try:
            self.ceos_corrector = CorrectorCommands(rpchost, rpcport)
            self.ceos_corrector.getInfo()
            print('Connected')
        except ConnectionRefusedError:
            print('Could not connect to RPC gateway')
            exit()
    
    def change_aberration(self, name, value, select):
        '''
        Change aberrations
        
        Parameters
        ----------
        name : str
            Name of aberration to be changed ('C1', 'A1', 'B2', 'A2', 'C3', 'A3', 'S3', 'We' (beam shift))
        value : 2-element list or array
            Amount to change the aberration by (in metres)
        select : str
            Use 'coarse' or 'fine' correction ('None' for C1 and C3)
        '''
        self.ceos_corrector.correctAberration(name=name, value=value, select=select)

class BEACON_Server():
    def __init__(self, port, rpchost, rpcport, SIM=False, TEST=False):
        
        self.SIM = SIM
        if not self.SIM:
            self.corrector = CEOS_RPC_control(rpchost, rpcport) 
            self.microscope = TIA_control()
        
        context = zmq.Context()
        serverSocket = context.socket(zmq.REP)
        serverSocket.bind('tcp://*:'+str(port))
        print('Server Online')
        
        self.refImage = None
        
        if TEST:
            self.d = {'type': 'ac',
                      'ab_values': {'C1': 0.0},
                      'ab_select': {'C1': None},
                      'dwell': 1e-7,
                      'shape': (256, 256),
                      'offset': (0, 0),
                      'metric': 'var',
                      'C1_defocus_flag': True,
                      'return_images': False,
                      'bscomp': False,
                      'ccorr': False,
                      }
            qval = self.acquire_image_with_aberrations()
            print(qval)

        while True:
            data = serverSocket.recv()
            self.d = pickle.loads(data)
            instruction = self.d['type']
            print(instruction)
            
            if instruction == 'ping':
                reply_message = 'pinged'
                reply_data = None
            elif instruction == 'ac':
                reply_message = 'ac'
                reply_data = self.acquire_image_with_aberrations()
            elif instruction == 'ref':
                self.refImage, _, _, _ = self.microscope.microscope_acquire_image(self.d['dwell'], self.d['shape'])
                reply_message = 'reference image set'
                reply_data = self.refImage
            elif instruction == 'ab_only':
                self.abChange(self.d['ab_values'], self.d['ab_select'], self.d['C1_defocus_flag'],
                              undo=False, bscomp=self.d['bscomp'])
                reply_message = 'aberrations changed'
                reply_data = None
            elif instruction == 'tableau':
                reply_message = 'tableau measured'
                reply_data = self.tableau_measurement()
            elif instruction == 'image':
                reply_message = 'image acquired'
                reply_data = self.microscope.microscope_acquire_image_old(self.d['dwell'], self.d['shape'], self.d['offset'])
            elif instruction == 'move_stage':
                print(self.d)
                self.microscope.move_stage_delta(self.d['dX'], self.d['dY'], self.d['dZ'], self.d['dA'], self.d['dB'])
                reply_message = 'stage moved'
                reply_data = None
            elif instruction == 'get_mag':
                reply_message = 'mag obtained'
                reply_data = self.microscope.get_mag()
            elif instruction == 'get_stage_pos':
                reply_message = 'pos obtained'
                reply_data = self.microscope.get_stage_pos()
            elif instruction == 'set_mag':
                self.microscope.set_mag(self.d['mag'])
                reply_message = 'mag changed'
                reply_data = None
            elif instruction == 'open_column_valve':
                self.microscope.open_column_valve()
                if self.microscope._microscope.Vacuum.ColumnValvesOpen:
                    reply_message = 'column valve open'
                else:
                    reply_message = 'column valve NOT open'
                reply_data = None
            elif instruction == 'close_column_valve':
                self.microscope.close_column_valve()
                if not self.microscope._microscope.Vacuum.ColumnValvesOpen:
                    reply_message = 'column valve closed'
                else:
                    reply_message = 'column valve NOT closed'
                reply_data = None
            else:
                reply_message = None
                reply_data = None
            
            reply_d = {'reply_message': reply_message,
                       'reply_data': reply_data}
            
            serverSocket.send(pickle.dumps(reply_d))
    
    def abChange(self, ab_values, ab_select, C1_defocus_flag, undo=False, bscomp=False):
        ''' 
        Change the aberrations
        
        Parameters
        ----------
        ab_values : dict
            Dictionary of aberration names and magnitudes that need to be changed.
        ab_select : dict
            Dictionary of aberration names and whether to use coarse or fine correction.
        C1_defocus_flag : bool
            True: Use microscope defocus to correct C1.
            False: Use aberration corrector to correct C1.
        undo : bool
            Apply the negative of ab_vals to change the aberrations
        bscomp : bool
            Use beam shift to compensate for changes in field of view when changing aberrations.
        '''
        
        ab_keys = list(ab_values.keys())
        ab_vals = list(ab_values.values())
        
        if undo:
            for i in range(len(ab_vals)):
                ab_vals[i] = -ab_vals[i]
        
        for i in range(len(ab_values)):
            if len(ab_keys[i])==2:
                if ab_keys[i] == 'C1':
                    if C1_defocus_flag:
                        self.microscope.change_defocus(ab_vals[i])
                    else:
                        self.corrector.change_aberration(name=ab_keys[i], value=[ab_vals[i],0], select=ab_select[ab_keys[i]])
                else:
                    self.corrector.change_aberration(name=ab_keys[i], value=[ab_vals[i],0], select=ab_select[ab_keys[i]])
            elif ab_keys[i].endswith('_x'):
                self.corrector.change_aberration(name=ab_keys[i][:2], value=[ab_vals[i],0], select=ab_select[ab_keys[i]])
            elif ab_keys[i].endswith('_y'): # UGLY!!!!
                self.corrector.change_aberration(name=ab_keys[i][:2], value=[0,ab_vals[i]], select=ab_select[ab_keys[i]])
        
        if bscomp:
            comp_x, comp_y = self.comp_shift_calc(ab_values)
            self.corrector.change_aberration(name='We', value=[comp_x, comp_y], select=None)
    
    def comp_shift_calc(self, ab_values):
        '''
        Calculates the value of beam shift by which compensate the aberrations
        N.B. The ssf_dict values will vary between microscopes. These were empirically determined 
            for the TEAM 0.5 using 320kx, 512x512 images, AuNPs, scan_rot=0
        NOT RECOMMENDED FOR GENERAL USE!!!
        
        Parameters
        ----------
        ab_values : dict
            Dictionary of aberration names and magnitudes that need to be changed.

        Returns
        -------
        comp_x : float
            Value (in m) by which to shift the beam in x to compensate for aberration correction.
        comp_y : float
            Value (in m) by which to shift the beam in y to compensate for aberration correction.
        '''
        
        # shift scaling factors (ssf, pixels/nm)
        ssf_dict = {
                    'C1': (0, 0),
                    'A1_x': (0, 0),
                    'A1_y': (0, 0),
                    'B2_x': (2/400, 0/400),
                    'B2_y': (3/400, -1/400),
                    'A2_x': (-5/400, -1/400),
                    'A2_y': (-7/400, -7/400),
                    'C3': (0, 0),
                    'A3_x': (-16/400, -4/400),
                    'A3_y': (2/400, -24/400),
                    'S3_x': (5/1000, 1/1000),
                    'S3_y': (0/1000, 5/1000),
                    }
        
        We_x_ssf = 21/10 # only has x-component
        We_y_ssf = -21/10 # only has y-component
        
        ab_list = list(ab_values.keys())
        shifts = np.zeros((len(ab_list), 2))
        
        for i, ab in enumerate(ab_list):
            shifts[i,0] = ab_values[ab]*ssf_dict[ab][0]
            shifts[i,1] = ab_values[ab]*ssf_dict[ab][1]
        
        shift_x = np.sum(shifts[:,0])
        shift_y = np.sum(shifts[:,1])
        
        comp_x = -shift_x/We_x_ssf
        comp_y = -shift_y/We_y_ssf
    
        return comp_x, comp_y

    def block_reduce_mean(self, image, block_size=(1, 1)):
        '''
        Mean pools the image by a given block size
        
        Parameters
        ----------
        image : array
            Image
        block_size : tuple or array
            Block shape / size over which to pool the image
            
        Returns
        -------
        reshaped image : array
            Pooled image
        '''
        
        b = block_size[0]
        s = image.shape[0]//b
        return image.reshape((s, b, s, b)).mean(axis=3).mean(axis=1)

    def correlate_func(self, im0, im1):
        '''
        Cross-correlate two images
        
        Parameters
        ----------
        im0 : array
            1st image
        im1 : array
            2nd image
            
        Returns
        -------
        Cross-correlation value
        '''

        p0 = np.zeros((im0.shape[0], im0.shape[1]))
        p1 = np.zeros((im0.shape[0], im0.shape[1]))
        p1[p1.shape[0]//2-im1.shape[0]//2:p1.shape[0]//2-im1.shape[0]//2 + im1.shape[0],
           p1.shape[1]//2-im1.shape[1]//2:p1.shape[1]//2-im1.shape[1]//2 + im1.shape[1]] = im1
        f0 = np.fft.fft2(im0)
        f1 = np.fft.fft2(p1)
        f0 *= np.conj(f1)
        c = np.fft.ifft2(f0)
        return np.fft.fftshift(c.real)

    def corr_cutout(self, cur_image, ref_image=None, brm=1):
        '''
        Cross-correlate two images and cut out the overlapping regions
        
        Parameters
        ----------
        cur_image : array
            Most recently acquired image
        ref_image : array
            Reference image
        brm : int
            Block reduce (binning) factor for use in cross-correlation
            
        Returns
        -------
        cutout : array
            Region of cur_image that overlaps with ref_image
        '''
        if ref_image is None:
            ref_image = self.refImage
        
        refIm = self.block_reduce_mean(ref_image, (brm,brm))
        curIm = self.block_reduce_mean(cur_image, (brm,brm))
        
        corr = self.correlate_func(curIm-curIm.mean(), refIm-refIm.mean())
        corr_arg = np.array(np.unravel_index(np.argmax(corr), corr.shape))
        offset = (corr_arg-np.array(refIm.shape)/2)
        
        x_start = int(np.array(refIm.shape[0])/4+offset[0])*brm
        x_end = int(3*np.array(refIm.shape[0])/4+offset[0])*brm
        y_start = int(np.array(refIm.shape[1])/4+offset[1])*brm
        y_end = int(3*np.array(refIm.shape[1])/4+offset[1])*brm
        
        cutout = cur_image[x_start:x_end,y_start:y_end]
        
        return cutout

    def metric_func(self, image_data, metric):
        '''
        Calculate quality metric. Current options are:
            Defocus Slice (df_slice) (for 1D defocus slices)
            Standard Deviation (std)
            Variance (var)
            Normalised Variance (normvar)
            Roughness (roughness)
        
        Parameters
        ----------
        image_data : array
            Image
        metric : str
            Quality metric ('df_slice' (for defocus), 'std', 'var', 'normvar', 'roughness')
            
        Returns
        -------
        qval : float
            Value of quality metric
        '''
        if not type(metric) is str:
            raise TypeError('Metric is not a string')
        if metric == 'df_slice':
            y = np.sum(image_data, axis=np.argmin(image_data.shape))
            fft_im = np.fft.fft(y)
            fft_abs = np.abs(fft_im)
            qval = np.sum(fft_abs[1:len(y)])/fft_abs[0]
        elif metric == 'std':
            qval = np.std(image_data)
        elif metric == 'var':
            qval = np.var(image_data)
        elif metric == 'normvar':
            qval = np.var(image_data)/(np.mean(image_data)**2)
        elif metric == 'roughness':
            #Calculate the Fourier coordinates with
            kx = np.fft.fftfreq(image_data.shape[0])
            ky = np.fft.fftfreq(image_data.shape[1])
            kr2 = kx[:,None]**2 + ky[None,:]**2
            #(or use np.outer for the outer product)
            
            #And an optional window function
            wx = np.hanning(image_data.shape[0])
            wy = np.hanning(image_data.shape[1])
            w = wx[:,None] * wy[None,:]
            #(or use np.outer for the outer product)
            
            #Then calculate the FFT intensity:
            G2 = np.abs(np.fft.fft2(image_data * w))**2
            #or
            #G2 = np.abs(np.fft.fft2(image_data))**2
            
            #And finally image roughness r2 using:
            r2 = np.sum(G2 * kr2 ) / np.sum(G2)
            qval = r2
        else:
            qval = None
        return qval
              
    def acquire_image_with_aberrations(self):
        '''
        Takes image with a given aberration (information contained in self.d dictionary) and returns the image
        
        Returns
        -------
        qval : float
            Quality metric.
        im_dict : dict
            Dictionary containing the image, calX and calY, unit name.
        '''
        
        if self.d is None:
            ab_values = {'C1': 0.0}
            ab_select = {'C1': None}
            dwell_time = 1e-7
            shape = (512, 512)
            offset = (0,0)
            metric = 'var'
            C1_defocus_flag = False
            return_images = False
            bscomp = False
            ccorr = False
        else:
            ab_values = self.d['ab_values']
            ab_select = self.d['ab_select']
            dwell_time = self.d['dwell']
            shape = self.d['shape']
            offset = self.d['offset']
            metric = self.d['metric']
            C1_defocus_flag = self.d['C1_defocus_flag']
            return_images = self.d['return_images']
            bscomp = self.d['bscomp']
            ccorr = self.d['ccorr']
        
        self.abChange(ab_values, ab_select, C1_defocus_flag, bscomp=bscomp)
        image_data, calX, calY, unitName = self.microscope.microscope_acquire_image(dwell_time, shape, offset)
        self.abChange(ab_values, ab_select, C1_defocus_flag, undo=True, bscomp=bscomp)
            
        if ccorr and shape[0]==shape[1]: # NEED TO CONSIDER HOW TO MAKE THIS WORK FOR NON-SQUARE IMAGE
            image = self.corr_cutout(image_data)
        elif ccorr and shape[0]!=shape[1]:
            image = image_data
            print('Cross-correlation not yet implemented for non-square images')
        else:
            image = image_data
        
        qval = self.metric_func(image, metric)
        im_dict = {'image': image,
                   'calX': calX,
                   'calY': calY,
                   'unitName': unitName}
        
        if return_images:
            return qval, im_dict
        else:
            return qval
    
    def tableau_measurement(self):
        '''
        Takes C1A1 measurement at a given beam tilt (information contained in self.d dictionary) and returns the C1A1
        
        Returns
        -------
        c1a1 : float
            C1/A1 measurements.
        '''
        
        if self.d is None:
            ab_values = {'WD_x': 0e-3,
                         'WD_y': 0e-3}
        else:
            ab_values = self.d['ab_values']
        
        print(ab_values)
        
        self.corrector.change_aberration(name='WD', value=[ab_values['WD_x'], ab_values['WD_y']], select=None)
        self.microscope.unblank()
        c1a1 = self.corrector.ceos_corrector.measureC1A1()
        self.microscope.blank()
        self.corrector.change_aberration(name='WD', value=[-ab_values['WD_x'], -ab_values['WD_y']], select=None)
        
        c1a1_dict = json.loads(c1a1[0].decode('utf-8'))['result']['aberrations']
        
        print(c1a1_dict)
        
        return c1a1_dict
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--serverport', action='store', type=int, default=7001, help='server port')
    parser.add_argument('--rpchost', action='store', type=str, default='localhost', help='rpc host')
    parser.add_argument('--rpcport', action='store', type=int, default=7072, help='rpc port')
    
    args = parser.parse_args()
    
    serverport = args.serverport
    rpchost = args.rpchost
    rpcport = args.rpcport
    
    server = BEACON_Server(serverport, rpchost, rpcport)