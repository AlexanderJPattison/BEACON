# BEACON (Bayesian-Enhanced Aberration Correction and Optimization Network)

# Requirements
Microscope computer
- CEOS RPC Gateway
- Python 3.4.3 or greater
- Python modules: numpy, pyzmq, pickle, socket, json, pynetstring

BEACON computer
- Python 3.9 or greater
- Python modules: gpcam 4.0.0, numpy, sys, pickle, pyzmq, matplotlib, pyqt

# Installation
BEACON uses a client-server model that allows the Bayesian optimization to be handled by a different computer to the one controlling the microscope (they can also be run on the same computer). You will need to edit the host and port names in GUI_client.py on lines 76 and 77 appropriately. If the CEOS RPC Gateway is being run on a different computer to the BEACON server, you will need to include the host and port names in the initialization of CorrectorCommands() in line 229 of Server.py.

1) Clone or download this repository
2) Move Server.py to the computer controlling the microscope and the CEOS corrector (Microscope computer)
3) Move GUI_Client.py to the computer that you wish to run the Bayesian optimization on (BEACON computer)

# Running BEACON
1) Open the CEOS RPC Gateway on your microscope computer
2) Run Server.py on your microscope computer
3) Run GUI_Client.py wherever it is installed
