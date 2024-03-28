# BO-for-AutoSTM


. Scanning Tunneling Microscopy (STM) is used to acquire atomically resolved images of material surfaces. The optimization of the imaging parameters is tedious due to the sensitive nature of the imaging method. Here we implement Bayesian Optimization (BO) method for autonomous tuning and convergence of the STM control parametrs. The publication describing the scientific results can be found at: doi.org/10.1063/5.0185362

The program "B_SP_prediction_using_BO", originally authored by Dr. Arpan Biswas, is used for the control parameter prediction for STM. The function "parameter_prediction()" is used to predict the next aquisition point of the contol parameters: Bias (V) and Setpoint(pA)
The demonstrative routine is provided at the end of the program in "BO_on_STM_toy_model.ipynb", where the fft_peak is extracted from the original data while the control paramters (i.e., bias and setpoint) are estimated using the BO algorithm.
