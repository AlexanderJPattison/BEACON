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
    
class TIA_control():
    def __init__(self):
        # Connect to the microscope
        self._microscope = CreateObject('TEMScripting.Instrument')
        self.TIA = CreateObject('ESVision.Application')
        self.Acq = self._microscope.Acquisition
        self.Ill = self._microscope.Illumination
        self.Proj = self._microscope.Projection
        
        # Connect to STEM
        detector0 = self.Acq.Detectors(0)
        # Add the first detector
        self.Acq.AddAcqDevice(detector0)

    def blank(self):
        ''' Blanks beam '''
        self.Ill.BeamBlanked = True
    
    def unblank(self):
        ''' Unblanks beam '''
        self.Ill.BeamBlanked = False
    
    def change_defocus(self, df):
        '''
        Changes the defocus
        
        Parameters
        ----------
        df : float
            Amount of defocus to change (in metres)
        '''
        currentDF = self.Proj.Defocus
        self.Proj.Defocus = currentDF + df

    def set_acquisition_parameters(self, binning, imsize, dwell_time):
        '''
        Sets image acquisition parameters
        
        Parameters
        ----------
        binning : int
            Binning level for image (4 for 1024, 8 for 512)
        imsize : int
            Size of image (0 = full size, 1 = half size, 2 = quarter size)
        dwell_time : float
            Dwell time (in seconds)
        '''

        myStemSearchParams = self.Acq.Detectors.AcqParams
        myStemSearchParams.Binning = binning
        myStemSearchParams.ImageSize = imsize
        myStemSearchParams.DwellTime = dwell_time
        self.Acq.Detectors.AcqParams = myStemSearchParams
    
    # NOT SURE WHETHER THIS IS MANDATORY OR NOT?
    def get_image_parameters(self):
        '''
        Get image parameters
        
        Returns
        -------
        unitName : str
            Units of the calibration (e.g. nm, um)
        calX : float
            x-calibration (nm per pixel)
        calY : float
            y-calibration (nm per pixel)
        '''
        
        window1 = self.TIA.ActiveDisplayWindow()
        Im1 = window1.FindDisplay(window1.DisplayNames[0]) #returns an image display object
        unit1 = Im1.SpatialUnit #returns SpatialUnit object
        unitName = unit1.unitstring #returns a string (such as nm)
        calX = self.TIA.ScanningServer().ScanResolution
        calY = self.TIA.ScanningServer().ScanResolution
        
        return unitName, calX, calY
    
    def microscope_acquire_image(self):
        '''
        Acquire image in TIA
        
        Returns
        -------
        image_data : array
            Acquired image
        '''
        if self.TIA.AcquisitionManager().isAcquiring:
            self.TIA.AcquisitionManager().Stop()
        self.unblank()
        # Acquire an image
        acquiredImageSet = self.Acq.AcquireImages()
        with safearray_as_ndarray:
            image_data = acquiredImageSet(0).AsSafeArray # get data as ndarray
        self.blank()
        
        return image_data

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
    def __init__(self, port, rpchost, rpcport, sim=False):
        
        self.SIM = sim
        if not self.SIM:
            self.corrector = CEOS_RPC_control(rpchost, rpcport) 
            self.microscope = TIA_control()
        
        context = zmq.Context()
        serverSocket = context.socket(zmq.REP)
        serverSocket.bind('tcp://*:'+str(port))
        print('Server Online')
        
        self.refImage = None
        
        ab_values = {'C1': 0.0}
        ab_select = {'C1': None}
        dwell_time = 1e-7
        size = 512
        metric = 'var'
        C1_defocus_flag = False
        return_images = False
        bscomp = False
        ccorr = False
        
        self.d = {'type': 'ac',
                  'ab_values': ab_values,
                  'ab_select': ab_select,
                  'dwell': dwell_time,
                  'size': size,
                  'metric': metric,
                  'C1_defocus_flag': C1_defocus_flag,
                  'return_images': return_images,
                  'bscomp': bscomp,
                  'ccorr': ccorr,
                  }
        
        TEST = False
        if TEST:
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
                self.refImage, _, _, _ = self.acquire_image(self.d['dwell'], self.d['size'])
                reply_message = 'reference image set'
                reply_data = self.refImage
            elif instruction == 'ab_only':
                self.abChange(self.d['ab_values'], self.d['ab_select'], self.d['C1_defocus_flag'],
                              undo=False, bscomp=self.d['bscomp'])
                reply_message = 'aberrations changed'
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

    def correlate_func(self, im0, im1, mode='full'):
        '''
        Cross-correlate two images
        
        Parameters
        ----------
        im0 : array
            1st image
        im1 : array
            2nd image
        mode : str
            Select which mode by which to perform cross-correlation
            
        Returns
        -------
        Cross-correlation value
        '''
        if mode == 'full':
            p0 = np.zeros((im0.shape[0] + im1.shape[0], im0.shape[1] + im1.shape[1]))
            p1 = np.zeros((im0.shape[0] + im1.shape[0], im0.shape[1] + im1.shape[1]))
            p0[p0.shape[0]//2-im0.shape[0]//2:p0.shape[0]//2-im0.shape[0]//2 + im0.shape[0],
               p0.shape[1]//2-im0.shape[1]//2:p0.shape[1]//2-im0.shape[1]//2 + im0.shape[1]] = im0
            p1[p1.shape[0]//2-im1.shape[0]//2:p1.shape[0]//2-im1.shape[0]//2 + im1.shape[0],
               p1.shape[1]//2-im1.shape[1]//2:p1.shape[1]//2-im1.shape[1]//2 + im1.shape[1]] = im1
            f0 = np.fft.fft2(p0)
            f1 = np.fft.fft2(p1)
            f0 *= np.conj(f1)
            c = np.fft.ifft2(f0)
            return np.fft.fftshift(c.real)
        if mode == 'same':
            p0 = np.zeros((im0.shape[0], im0.shape[1]))
            p1 = np.zeros((im0.shape[0], im0.shape[1]))
            p1[p1.shape[0]//2-im1.shape[0]//2:p1.shape[0]//2-im1.shape[0]//2 + im1.shape[0],
               p1.shape[1]//2-im1.shape[1]//2:p1.shape[1]//2-im1.shape[1]//2 + im1.shape[1]] = im1
            f0 = np.fft.fft2(im0)
            f1 = np.fft.fft2(p1)
            f0 *= np.conj(f1)
            c = np.fft.ifft2(f0)
            return np.fft.fftshift(c.real)
        if mode == 'valid':
            p0 = np.zeros((im0.shape[0], im0.shape[1]))
            p1 = np.zeros((im0.shape[0], im0.shape[1]))
            p1[p1.shape[0]//2-im1.shape[0]//2:p1.shape[0]//2-im1.shape[0]//2 + im1.shape[0],
               p1.shape[1]//2-im1.shape[1]//2:p1.shape[1]//2-im1.shape[1]//2 + im1.shape[1]] = im1
            f0 = np.fft.fft2(im0)
            f1 = np.fft.fft2(p1)
            f0 *= np.conj(f1)
            c = np.fft.ifft2(f0)
            return np.fft.fftshift(c.real)[c.shape[0]//2-(im0.shape[0]-im1.shape[1])//2:c.shape[0]//2-(im0.shape[0]-im1.shape[0])//2+im0.shape[0]-im1.shape[0],
                                           c.shape[1]//2-(im0.shape[1]-im1.shape[1])//2:c.shape[1]//2-(im0.shape[1]-im1.shape[1])//2+im0.shape[1]-im1.shape[1]]

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
        
        corr = self.correlate_func(curIm-curIm.mean(), refIm-refIm.mean(), mode='same')
        corr_arg = np.array(np.unravel_index(np.argmax(corr), corr.shape))
        offset = (corr_arg-np.array(refIm.shape)/2)
        
        #print(offset)
        
        x_start = int(np.array(refIm.shape[0])/4+offset[0])*brm
        x_end = int(3*np.array(refIm.shape[0])/4+offset[0])*brm
        y_start = int(np.array(refIm.shape[1])/4+offset[1])*brm
        y_end = int(3*np.array(refIm.shape[1])/4+offset[1])*brm
        
        cutout = cur_image[x_start:x_end,y_start:y_end]
        
        return cutout

    def metric_func(self, image_data, metric):
        '''
        Calculate quality metric. Current options are:
            Standard Deviation (std)
            Variance (var)
            Normalised Variance (normvar)
            Roughness (roughness)
        
        Parameters
        ----------
        image_data : array
            Image
        metric : str
            Quality metric ('std', 'var', 'normvar', 'roughness')
            
        Returns
        -------
        qval : float
            Value of quality metric
        '''
        if not type(metric) is str:
            raise TypeError('Metric is not a string')
        if metric == 'slice':
            slice_width = 5
            imsize = len(image_data)
            dfslice = image_data[int(imsize/2-slice_width):int(imsize/2+slice_width)]
            y = np.sum(dfslice, axis=0)
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
    
    def acquire_image(self, dwell_time, size):
        '''
        Acquire an image from the microscope
        
        Parameters
        ----------
        dwell_time : float
            Dwell time (in seconds).
        size : int
            Image size.

        Returns
        -------
        image_data : array
            Image.
        unitName : str
            Units of the calibration (e.g. nm, um)
        calX : float
            x-calibration (nm per pixel)
        calY : float
            y-calibration (nm per pixel)
        '''
        
        if size < 512:
            binning = 8
            imsize = int(np.log2(512)-np.log2(size))
        else:
            binning = int(4096/size)
            imsize = 0
        
        self.microscope.set_acquisition_parameters(binning, imsize, dwell_time)
        unitName, calX, calY = self.microscope.get_image_parameters()
        image_data = self.microscope.microscope_acquire_image()
        
        return image_data, unitName, calX, calY
            
    def acquire_image_with_aberrations(self):
        '''
        Takes image with a given aberration (information contained in self.d dictionary) and returns the image
        
        Returns
        -------
        qval : float
            Quality metric.
        im_dict : dict
            Dictionary containing the image, unit name, calX and calY.
        '''
        
        if self.d is None:
            ab_values = {'C1': 0.0}
            ab_select = {'C1': None}
            dwell_time = 1e-7
            size = 512
            metric = 'var'
            C1_defocus_flag = False
            return_images = False
            bscomp = False
            ccorr = False
        else:
            ab_values = self.d['ab_values']
            ab_select = self.d['ab_select']
            dwell_time = self.d['dwell']
            size = self.d['size']
            metric = self.d['metric']
            C1_defocus_flag = self.d['C1_defocus_flag']
            return_images = self.d['return_images']
            bscomp = self.d['bscomp']
            ccorr = self.d['ccorr']
        
        self.abChange(ab_values, ab_select, C1_defocus_flag, bscomp=bscomp)
        image_data, unitName, calX, calY = self.acquire_image(dwell_time, size)
        self.abChange(ab_values, ab_select, C1_defocus_flag, undo=True, bscomp=bscomp)
            
        if ccorr:
            image = self.corr_cutout(image_data)
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