# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:35:09 2023

@author: advanced_user
"""
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from GUI_Client import BEACON_Client

parser = argparse.ArgumentParser()
parser.add_argument('--serverhost', action='store', type=str, default='localhost', help='server host')
parser.add_argument('--serverport', action='store', type=int, default=7001, help='server port')

args = parser.parse_args()

host = args.serverhost
port = args.serverport

ac_ae = BEACON_Client(host, port)

range_dict = {#'C1': [-100,100],
              'A1_x': [-100,100],
              'A1_y': [-100,100],
              #'B2_x': [-300,300],
              #'B2_y': [-300,300],
              #'A2_x': [-300,300],
              #'A2_y': [-300,300],
              #'C3':[-500,500],
              #'S3_x': [-1000,1000],
              #'S3_y': [-1000,1000],
              #'A3_x': [-3000,3000],
              #'A3_y': [-3000,3000],
              }

ab_select = {'C1': None,
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

init_size_value = 5
runs_value = 40
func_value = 'ucb'

dwell_value = 3e-6
size_value = 512
metric_value = 'normvar'

return_images = True
bscomp = False
ccorr = True

for n in range(1):
    print(n)
    ac_ae.ae_main(range_dict,
                  init_size_value, 
                  runs_value,
                  func_value,
                  dwell_value, 
                  size_value, 
                  metric_value,
                  return_images,
                  bscomp,
                  ccorr,
                  C1_defocus_flag=True,
                  return_dict=True,
                  return_all_f_re=True,
                  return_final_f_re=True,
                  return_model_max_list=True,
                  ab_select=None,
                  custom_early_stop_flag=False,
                  custom_ucb_factor=3,
                  noise_level=1e-4,
                  init_hps=None,
                  hp_bounds=None)

fig, ax = plt.subplots(1,2)
ax[0].imshow(ac_ae.BEACON_dict['initial_image']['image'])
ax[1].imshow(ac_ae.BEACON_dict['final_image']['image'])

mm = ac_ae.model_max
ab_keys = ac_ae.ab_keys
ab_values = {}
for i in range(len(ab_keys)):
    ab_values[ab_keys[i]] = mm[i]*1e-9

x = input('Correct? 0/1')
if x == '1':
    print('Correcting')
    ac_ae.ab_only(ab_values)
