# modulation_classification

here I proposed the solution for the modulation classification competition. In this solution I used machine learning programming frameworks Tensorflow and Keras. 

Task 1.  Distinguish between BPSK and GFSK
Part1. Carry out the training separately for each SNR value, i.e., have one classifier for every SNR point
       For this problem I proposed the model using Deep Neural Network . For this model the training can be done on a laptop. 
Part2. Carry out the training jointly over all SNR values
       For this problem I proposed the model using 1 Dimensional Convolutional Neural Network.  The training process is done on google             colab   server using the provided GPU. 

Task 2. Classify modulation format 
The goal is to classify between 10 of the modulation formats, and the performance is measured according its SNR value. 
Part1. Carry out the training separately for each SNR value, i.e., have one classifier for every SNR point
       For this problem I proposed the model using Deep Neural network
Part2.  Carry out the training jointly over all SNR values
      For this problem I proposed the model using 1 Dimensional Convolutional Neural Network. The training process is done on google colab       server using the provided GPU. One epochs roughly took 15 seconds to run. there are 100 epochs.  



