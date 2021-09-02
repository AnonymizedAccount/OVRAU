# OVRAU 
An **O**verall low-dimensional **V**ector **R**epresentations for **A**nchor **U**sers on multiplex heterogeneous social networks
## Overview 
This directory contains code necessary to run the OVRAU algorithm. OVRAU can be viewed as a graph convolutional neural network based on three candidate aggregator functions incuding mean aggegator, max-pooling aggregator, and LSTM aggregator. It is especially useful for learning latent representations for shared or anchor users across social networks to capture their intra-network and cross network structural information.
## Requirements
- Python >= 3.7
- TensorFlow >= 2.2.0

After verifying the installed versions of Python and TensorFlow, then install other dependencies by

   ```pip install -r requirements.txt```
## Running the code 

After installing the required dependencies, clone this repository using the following command:

  ```
  git clone https://github.com/AnonymizedAccount/OVRAU
  cd OVRAU 
  ```
