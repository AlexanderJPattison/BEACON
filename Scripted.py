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

start = time.time()

ac_ae = BEACON_Client(host, port)

range_dict = {'C1': [-10,10],
              'A1_x': [-10,10],
              'A1_y': [-10,10],
              'B2_x': [-300,300],
              'B2_y': [-300,300],
              'A2_x': [-300,300],
              'A2_y': [-300,300],
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

init_size_value = 20
runs_value = 130
func_value = 'ucb'

dwell_value = 3e-6
shape_value = (256,256)
offset_value = (0,0)
metric_value = 'normvar'

return_images = True
bscomp = False
ccorr = True

mm_array = np.zeros((5,len(range_dict)))

for ii in range(len(mm_array)):
    ac_ae.ae_main(range_dict,
                  init_size_value, 
                  runs_value,
                  func_value,
                  dwell_value,
                  shape_value,
                  offset_value,
                  metric_value,
                  return_images,
                  bscomp,
                  ccorr,
                  C1_defocus_flag=True,
                  include_norm_runs=False,
                  ab_select=ab_select,
                  return_dict=True,
                  return_all_f_re=False,
                  return_final_f_re=False,
                  return_model_max_list=True,
                  custom_early_stop_flag=False,
                  ucb_coefficient=2,
                  noise_level=0.1,
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
    
    mm_array[ii] = mm

'''
x = input('Correct? 0/1')
if x == '1':
    print('Correcting')
    ac_ae.ab_only(ab_values)
'''
'''
end = time.time()
time_taken = end-start

print(time_taken)
'''

mm_mag = np.linalg.norm(mm_array, axis=1)
mm_mag_mean = np.mean(mm_mag)
mm_mag_std = np.std(mm_mag)

print(f'{mm_mag_mean}, {mm_mag_std}')

#A1: 1.63011585, 1.74120843
#%%
'''
A1_mm_mag = np.linalg.norm(A1_mm_array, axis=1)
A1_mm_mag_mean = np.mean(A1_mm_mag)
A1_mm_mag_std = np.std(A1_mm_mag)

print(f'{A1_mm_mag_mean}, {A1_mm_mag_std}')

B2_mm_mag = np.linalg.norm(B2_mm_array, axis=1)
B2_mm_mag_mean = np.mean(B2_mm_mag)
B2_mm_mag_std = np.std(B2_mm_mag)

print(f'{B2_mm_mag_mean}, {B2_mm_mag_std}')

A2_mm_mag = np.linalg.norm(A2_mm_array, axis=1)
A2_mm_mag_mean = np.mean(A2_mm_mag)
A2_mm_mag_std = np.std(A2_mm_mag)

print(f'{A2_mm_mag_mean}, {A2_mm_mag_std}')

B2_A2_mm_array_B2 = mm_array[:,:2]
B2_A2_mm_array_A2 = mm_array[:,2:]

B2_A2_mm_mag_B2 = np.linalg.norm(B2_A2_mm_array_B2, axis=1)
B2_A2_mm_mag_mean_B2 = np.mean(B2_A2_mm_mag_B2)
B2_A2_mm_mag_std_B2 = np.std(B2_A2_mm_mag_B2)

B2_A2_mm_mag_A2 = np.linalg.norm(B2_A2_mm_array_A2, axis=1)
B2_A2_mm_mag_mean_A2 = np.mean(B2_A2_mm_mag_A2)
B2_A2_mm_mag_std_A2 = np.std(B2_A2_mm_mag_A2)

print(f'{B2_A2_mm_mag_mean_B2}, {B2_A2_mm_mag_std_B2}')
print(f'{B2_A2_mm_mag_mean_A2}, {B2_A2_mm_mag_std_A2}')

36.60387882485541, 27.948055272744707
6.6148280935806785, 4.251670697242646

C1: 1.8992839577349403, 0.01207764180947354 # 1.9 MAY BE OPTIMAL

mm_array = mm_array[mm_array[:,0]>-5]

C1_A1_mm_array_C1 = mm_array[:,:2]
C1_A1_mm_array_A1 = mm_array[:,2:]

C1_A1_mm_mag_C1 = np.linalg.norm(C1_A1_mm_array_C1, axis=1)
C1_A1_mm_mag_mean_C1 = np.mean(C1_A1_mm_mag_C1)
C1_A1_mm_mag_std_C1 = np.std(C1_A1_mm_mag_C1)

C1_A1_mm_mag_A1 = np.linalg.norm(C1_A1_mm_array_A1, axis=1)
C1_A1_mm_mag_mean_A1 = np.mean(C1_A1_mm_mag_A1)
C1_A1_mm_mag_std_A1 = np.std(C1_A1_mm_mag_A1)

print(f'{C1_A1_mm_mag_mean_C1}, {C1_A1_mm_mag_std_C1}')
print(f'{C1_A1_mm_mag_mean_A1}, {C1_A1_mm_mag_std_A1}')

1.933628589019686, 0.48999832631535195
0.5448859702837017, 0.7268362908407474


SevenD_mm_array_C1 = mm_array[:,:2]
SevenD_mm_array_A1 = mm_array[:,2:4]
SevenD_mm_array_B2 = mm_array[:,4:6]
SevenD_mm_array_A2 = mm_array[:,6:]

SevenD_mm_mag_C1 = np.linalg.norm(SevenD_mm_array_C1, axis=1)
SevenD_mm_mag_mean_C1 = np.mean(SevenD_mm_mag_C1)
SevenD_mm_mag_std_C1 = np.std(SevenD_mm_mag_C1)

SevenD_mm_mag_A1 = np.linalg.norm(SevenD_mm_array_A1, axis=1)
SevenD_mm_mag_mean_A1 = np.mean(SevenD_mm_mag_A1)
SevenD_mm_mag_std_A1 = np.std(SevenD_mm_mag_A1)

SevenD_mm_mag_B2 = np.linalg.norm(SevenD_mm_array_B2, axis=1)
SevenD_mm_mag_mean_B2 = np.mean(SevenD_mm_mag_B2)
SevenD_mm_mag_std_B2 = np.std(SevenD_mm_mag_B2)

SevenD_mm_mag_A2 = np.linalg.norm(SevenD_mm_array_A2, axis=1)
SevenD_mm_mag_mean_A2 = np.mean(SevenD_mm_mag_A2)
SevenD_mm_mag_std_A2 = np.std(SevenD_mm_mag_A2)

print(f'{SevenD_mm_mag_mean_C1}, {SevenD_mm_mag_std_C1}')
print(f'{SevenD_mm_mag_mean_A1}, {SevenD_mm_mag_std_A1}')
print(f'{SevenD_mm_mag_mean_B2}, {SevenD_mm_mag_std_B2}')
print(f'{SevenD_mm_mag_mean_A2}, {SevenD_mm_mag_std_A2}')


'''