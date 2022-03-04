# Photometric Redshift Estimation with Convolutional Neural Networks and Galaxy Images: Case Study of Resolving Biases in Data-Driven Methods

ArXiv: https://arxiv.org/abs/2202.09964

We investigate two forms of biases, i.e., "class-dependent residuals" and "mode collapse" in a case study of estimating photometric redshifts as a classification problem using Convolutional Neural Networks (CNNs) trained with galaxy images with spectroscopic redshifts. We propose a set of consecutive steps to resolve the two biases:

- Step 1: Representation learning with a multi-channel output module using all training data.
- Step 2 (To correct for overpopulation-induced $z_{spec}$-dependent residuals): Fix the representation and fine-tune the output module using a nearly balanced subset of training data.
- Step 3 (To correct for mode collapse and underpopulation-induced $z_{spec}$-dependent residuals): Fix the representation, extend the redshift output range, readjust the training labels, and retrain the extended output module using the same nearly balanced subset.
- Step 4 (To correct for $z_{photo}$-dependent residuals): Calibrate 

The code is tested using: 
- Python 2.7.15
- TensorFlow 1.12.0
- CPU: Intel(R) Core(TM) i9-7920X
- GPU: Titan V / GeForce RTX 2080 Ti

## Results


![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/image_examples.png)

![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/variant_analysis.png)


## Training and Testing

> python model.py --method=? --phase=train
