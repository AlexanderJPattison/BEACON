import matplotlib.pyplot as plt
import argparse
from beacon_client import BEACON_Client

parser = argparse.ArgumentParser()
parser.add_argument('--serverhost', action='store', type=str, default='localhost', help='server host')
parser.add_argument('--serverport', action='store', type=int, default=7001, help='server port')

args = parser.parse_args()

host = args.serverhost
port = args.serverport

beacon = BEACON_Client(host, port)

# Search boundaries (units of nm)
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

init_size_value = 5 # initial number of runs (pre Bayesian optimization)
runs_value = 25 # total number of runs
func_value = 'ucb' # 'ucb' : upper confidence bound method

dwell_value = 3e-6 # units of seconds
shape_value = (256,256) # image shape
offset_value = (0,0) # offset from centre
metric_value = 'normvar' # 'normvar' : normalized variance

return_images = True # return images from the server
bscomp = False # beam shift compensation (NOT recommended!)
ccorr = True # cross-correlation (strongly recommended)

# Run BEACON algorithm
beacon.ae_main(range_dict,
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

# Plot initial and final image
fig, ax = plt.subplots(1,2)
ax[0].imshow(beacon.BEACON_dict['initial_image']['image'])
ax[1].imshow(beacon.BEACON_dict['final_image']['image'])

# Extract model maximum (minimum aberration state)
mm = beacon.model_max
ab_keys = beacon.ab_keys
ab_values = {}
for i in range(len(ab_keys)):
    ab_values[ab_keys[i]] = mm[i]*1e-9

# Apply correction
x = input('Correct? 0/1')
if x == '1':
    print('Correcting')
    beacon.ab_only(ab_values)
