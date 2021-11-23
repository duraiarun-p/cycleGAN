#!/bin/bash

echo "Conda sourcing"
source /home/arun/anaconda3/etc/profile.d/conda.sh
echo "Conda restarted"
conda activate dlenv
echo "DLENV env activated"
echo "Launching CycleGAN training in DLENV environment"
InputPath="/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/"
OutputPath1="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/alpha"
OutputPath2="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/beta"
OutputPath3="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/gamma"
OutputPath4="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/delta"
OutputPath5="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/epsilon"
OutputPath6="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/zeta"
OutputPath7="/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/eta"
#python dblmcyclegan_beta.py -i /home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/ -o /home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/2992021/beta
python eddie_dblmcyclegan_beta.py -i $InputPath -o $OutputPath3
