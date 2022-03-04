# Photometric Redshift Estimation with Convolutional Neural Networks and Galaxy Images: Case Study of Resolving Biases in Data-Driven Methods

ArXiv: https://arxiv.org/abs/2202.09964

We investigate two forms of biases, i.e., "class-dependent residuals" and "mode collapse" in a case study of estimating photometric redshifts as a classification problem using Convolutional Neural Networks (CNNs) trained with galaxy images with spectroscopic redshifts. The data are from the Sloan Digital Sky Survey (SDSS) and the Canada–France–Hawaii Telescope Legacy Survey (CFHTLS).

We propose a set of consecutive steps to resolve the two biases:
- Step 1: Representation learning with a multi-channel output module using all training data.
- Step 2 (To correct for overpopulation-induced z_{spec}-dependent residuals): Fix the representation and fine-tune the output module using a nearly balanced subset of training data.
- Step 3 (To correct for mode collapse and underpopulation-induced z_{spec}-dependent residuals): Fix the representation, extend the redshift output range, readjust the training labels, and retrain the extended output module using the same nearly balanced subset.
- Step 4 (To correct for z_{photo}-dependent residuals): Calibrate z_{photo} in each (r-magnitude, z_{photo}) cell based on a resampled training set whose (r-magnitude, z_{spec}) distribution matches the (r-magnitude, z_{photo}) distribution of the test set (regarded as the "expected" (r-magnitude, z_{spec}) distribution).


## Results

![image](https://github.com/QiufanLin/PZbias/main/delz_compare_new2.png)

![image](https://github.com/QiufanLin/PZbias/main/SDSSresP_new2.png)


## Training and Testing

The code is tested using: 
-- Python 2.7.15
-- TensorFlow 1.12.0
-- CPU: Intel(R) Core(TM) i9-7920X
-- GPU: Titan V / GeForce RTX 2080 Ti


- Main experiments with the SDSS data and "Net_P"
* Both training and testing have to be run consecutively in either Baseline or Steps 1~3; this applies to all the following cases.

-- Baseline (training):
> python python PZbias_main.py --ne=1 --fth=0 --nsub=0 --errl=0 --midreduce=0 --testphase=0 --tstep=0 --multir=0 --usecfht=0 --usecfhtd=0 --bins=180 --net=0 --itealter=0 --softlabel=1 --shiftlabel=1

-- Baseline (testing):
> python python PZbias_main.py --ne=1 --fth=0 --nsub=0 --errl=0 --midreduce=0 --testphase=1 --tstep=0 --multir=0 --usecfht=0 --usecfhtd=0 --bins=180 --net=0 --itealter=0 --softlabel=1 --shiftlabel=1
(Set "--testphase=1")

-- Step 1 (training):
(Set "--tstep=1", "--multir=1", "--testphase=0")

-- Step 1 (testing):
(Set "--tstep=1", "--multir=1", "--testphase=1")

-- Step 2 (training):
(Set "--fth=200", "--tstep=2", "--multir=1", "--testphase=0")

-- Step 2 (testing):
(Set "--fth=200", "--tstep=2", "--multir=1", "--testphase=1")

-- Step 3 (training):
(Set "--fth=200", "--tstep=3", "--multir=1", "--testphase=0")

-- Step 3 (testing):
(Set "--fth=200", "--tstep=3", "--multir=1", "--testphase=1")
