# OVRAU 
An **O**verall low-dimensional **V**ector **R**epresentations for **A**nchor **U**sers on multiplex heterogeneous social networks
## Overview 
This directory contains code necessary to run the OVRAU algorithm. OVRAU can be viewed as a graph convolutional neural network based on three candidate aggregator functions incuding mean aggegator, max-pooling aggregator, and LSTM aggregator. It is especially useful for learning latent representations for shared or anchor users across social networks to capture their intra-network and cross network structural information.
## Requirements
- Python >= 3.7
- TensorFlow >= 2.2.0

## Installation 

Clone this repository:

  ```
  git clone https://github.com/AnonymizedAccount/OVRAU
  cd OVRAU 
  ```
Then, install other dependencies by

   ```
   pip install -r requirements.txt
   ```
## Dataset 
Due to the size limitation of the repository, we only provide few small dataset under the folder **data** to help you understand our code and reproduce our experiment. You are welcome to contact us to get access to the whole used dataset.
### Input format
You can also use your own multiplex social network dateset, as long as it fits the following template.

```
edge_type head tail weight
    r1     n1   n2    1
    r2     n2   n3    1
    .
    .
    .
```
