# BEACON (Bayesian-Enhanced Aberration Correction and Optimization Network)

![Intro Figure](https://github.com/user-attachments/assets/a95c1fe3-f906-45b1-90ea-ab05471ab26d)

# Requirements
At present, BEACON is only compatible with ThermoFisher microscopes as elements of Server.py use TIA scripting. These elements would need to be rewritten for microscopes from other vendors, but GUI_Client.py and Scripted.py would remain the same.

Microscope computer
- CEOS RPC Gateway
- Python 3.4.3 or greater
- Python modules: numpy, pyzmq, pickle, socket, json, pynetstring

BEACON computer
- Python 3.9 or greater
- Python modules: gpcam 4.0.0, numpy, sys, pickle, pyzmq, matplotlib, pyqt

# Installation
BEACON uses a client-server model that allows the Bayesian optimization to be handled by a different computer to the one controlling the microscope (they can also be run on the same computer).

1) Clone or download this repository.
2) Move Server.py to the computer controlling the microscope and the CEOS corrector (Microscope computer).
3) Move GUI_Client.py to the computer that you wish to run the Bayesian optimization on (BEACON computer).

# Opening BEACON
1) Open the CEOS RPC Gateway on your microscope computer.
2) On your microscope computer, run `BEACON_Server.bat' (Start menu on NCEM computers)

--OR--

2) On your microscope computer, run `python Server.py --serverport <IP port address for server> --rpchost <IP host address of CEOS RPC gateway> --rpcport <IP port address of CEOS RPC gateway>`.
3) On your BEACON computer, run `python GUI_Client.py --serverhost <IP host address of server> --serverport <IP port address of server>`.

--OR--

3) On your BEACON computer, run `python Scripted.py --serverhost <IP host address of server> --serverport <IP port address of server>`.
Note: Default IP addresses are: `--serverhost 'localhost', --serverport 7001, --rpchost 'localhost', --rpcport 7072`
Note: The `Connected to 'host' at 'port'' message on the client is meaningless. The way to determine if it has truly connected is to look at the server output and see `ping' after you have opened the client. I will fix this in a future update.

# Setting up BEACON
Click the checkboxes next to aberrations you want to correct and select the upper and lower search bounds for the optimization. These bounds are relative to the current state (i.e. 0 is the current value).

`Image Shape (x,y)`: shape of the images used in the optimization.

`Offset (x,y)`: Not currently implemented. This will allow you to offset the image from its current centre (particularly useful for defocus slices). This will be implemented in a future update.

`Dwell time (us)`; dwell time of the beam at every probe position.

`Metric`: metric that BEACON uses to optimize the image. Normalized variance, variance and standard deviation are functionally similar and can likely be used interchangeably (especially if `Use Cross Correlation' is checked). Defocus slice was developed for C1 and is only recommended for use with centrosymmetric aberrations (i.e. only C1).

`Initial Iterations`: number of iterations at the start of the optimization. Three of these will be normalization iterations acquired at the current state, the lower bound and the upper bound.

`Optimization Iterations`: number of optimization iterations using whichever method is selected.

`Extra Iterations`: number of additional iterations performed when `Continue' is clicked.

`Method`: acquisition function used to determine the next sample measurement (currently only Upper Confidence Bound is available).

`UCB Coefficient`: coefficient of the Upper Confidence Bound method. Increasing this number favors exploration (searching unexplored regions for the maximum) while decreasing this number favors exploitation (searching the explored regions for a quicker solution).

`Noise level`: signal-to-noise ratio for repeated measurements. THIS VARIABLE HAS THE MOST IMPACT ON PERFORMANCE. To get a rough idea, I would recommend trying to optimize one variable over the range that you deem appropriate, looking at the line graph and seeing what the spread is at a given measurement. Generally speaking, I would recommend ~0.01 for samples with well-defined edges (like nanoparticles) and ~0.1 for samples like thin films. This is by far the most difficult quantity to determine and I would welcome any input on how to make this system more user-friendly.

`Show Images`: Select if you want to see the images on the right during the optimization runs.

`Use Cross Correlation`: This determines whether cross-correlation is used to ensure the same field of view between scans. Generally, this should always be on. Only deselect if cross-correlation causes problems (does happen occasionally) or testing a new algorithm that is position invariant.

`Compensate with Beam Shift`: NOT RECOMMENDED. This was an alternative to cross-correlation for maintaining the field of view, which was required to test third-order aberrations. However, this appears to introduce extra aberrations that make this method worthless. Only use for testing.

# Running BEACON
Once all parameters have been set, click `Start` to begin.

`Stop` stops the optimization run mid-run. Suggested corrections can be accepted or rejected at this stage.

Once the run has finished, there are three options: `Accept`, which accepts the suggested corrections, `Reject`, which rejects the suggested corrections, and `Continue`, which performs extra iterations as set by `Extra Iterations`. Look at the initial and final image to determine if the optimization run has improved the image quality. If the quality metric in the `Status Box` is greater than 1, this indicates an improvement. A quality metric of less than 1 does not necessarily imply that the optimization is bad, as many factors can contribute to this. If you accept a correction and then wish to reverse it, click `Undo Last`.

NOTE: This is very much a product in development. Sensible inputs should not result in errors, but there are likely many potential ways to break this system. Please note that this software does not permanently change any aberration unless you click `Accept`, so do not be afraid to close and reopen the software if there are issues. Any feedback on usability would be greatly appreciated.
