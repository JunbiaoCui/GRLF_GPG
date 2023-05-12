# Code of "A General Representation Learning Framework with Generalization Performance Guarantees"

# 1. Fold Structure
## 1.1 /GRLF_GPG/Core/
The core function modules, including:

### (1) Radius.py
Code implementation of **Algorithm** 1 in **Appendix** B.1. 

### (2) Margin_DisConvexHull.py
Code implementation of **Algorithm** 2 in **Appendix** B.1.

### (3) VC_Dim_KernelSelection.py
Code implementation of **Algorithm** 3 in **Appendix** B.2.

### (4) VC_Dim_DNNBoostFramework.py
Code implementation of **Algorithm** 4 in **Appendix** B.3 and Fully-Connected Network.

### (5) StandNet.py
Code implementation of Standardization Network, a part of **Algorithm** 4. See formula (12)-->(13).

### (6) VC_Dim_Loss.py
Code implementation of VC dimension based loss, a part of **Algorithm** 4. See formula (14).

### (7) DataSet.py
Code implementation of some data processing modules, including:
load data from file and generate epoch, batch, etc.

## 1.2 Demos to illustrate how the proposed methods work, 
including:
### (1) Demo_KernelSelection.py
An experiment on Taichi data set, 628+629=1257 training samples, Gaussain kernel, 2000 kernel parameters.
	
### (2) Demo_DNNBoost.py
An experiment on MNIST data set, 10x10=100 training samples, 10000 test sample, FCNet.\
When VC_Dim_Loss_weight>0, the proposed boosting module will work.\
When VC_Dim_Loss_weight=0, the proposed boosting module will not work.

## 1.3 /GRLF_GPG/DataSet/
Two data sets, including:
### (1) /TaiChi/
Files of TaiChi data set.
	
### (2) /MNIST/
Files of MNIST data set.
	    
## 1.4 /GRLF_GPG/Result/
The default root folder of result files, including:
### (1) /TaiChi/
Result files of experiments on Taichi data set.
	
### (2) /MNIST/
Result files of experiments on MNIST data set.

## 1.5 ReadMe.md
This file.

# 2. Runtime Environment
## 2.1 OS (Operating System )
Ubuntu 18.04.5

## 2.2 IDE (Integrated Development Environment)
PyCharm 2020.1.5

## 2.3 Main Libraries
Python 3.8.12\
PyTorch 1.7.0\
Numpy 1.21.2\
hdf5storage 0.1.16\
scipy 1.7.1\
etc.
