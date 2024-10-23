# BEACON (Bayesian-Enhanced Aberration Correction and Optimization Network)

![Intro Figure](https://github.com/user-attachments/assets/a95c1fe3-f906-45b1-90ea-ab05471ab26d)

# Requirements
Microscope computer
- CEOS RPC Gateway
- Python 3.4.3 or greater
- Python modules: numpy, pyzmq, pickle, socket, json, pynetstring

BEACON computer
- Python 3.9 or greater
- Python modules: gpcam 4.0.0, numpy, sys, pickle, pyzmq, matplotlib, pyqt

# Installation
BEACON uses a client-server model that allows the Bayesian optimization to be handled by a different computer to the one controlling the microscope (they can also be run on the same computer).

1) Clone or download this repository
2) Move Server.py to the computer controlling the microscope and the CEOS corrector (Microscope computer)
3) Move GUI_Client.py to the computer that you wish to run the Bayesian optimization on (BEACON computer)

# Running BEACON
1) Open the CEOS RPC Gateway on your microscope computer
2) On your microscope computer, run `python Server.py --serverport <IP port address for server> --rpchost <IP host address of CEOS RPC gateway> --rpcport <IP port address of CEOS RPC gateway>`
3) On your BEACON computer, run `python GUI_Client.py --serverhost <IP host address of server> --serverport <IP port address of server>`
Note: Default IP addresses are: `--serverhost 'localhost', --serverport 7001, --rpchost 'localhost', --rpcport 7072`
