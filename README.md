This repository includes the source code for the paper "Whitening CNN-based rotor system fault diagnosis model features"

https://doi.org/10.3390/app12094411

The code directory contains the python files required to run the bearing fault tests.
Thruster data could not be shared due to legal restrictions. The code related to thruster tests is however available for
examination.

To run the bearing fault test:

1. cd original.tmp/12k
2. unzip drive_end2.zip
3. cd ..
4. cd ..
5. pip install -r requirements.txt
6. cd code
7. bash run_wdcnn_deconv.sh
8. bash run_[insert model].sh
