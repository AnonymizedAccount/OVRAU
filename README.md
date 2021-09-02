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
You can also use your own multiplex social network dateset, you should prepare the following three files (train.txt, test.txt, and valid.txt), as long as it fits the following template.

```
edge_type head tail weight
    r1     n1   n2    1
    r2     n2   n3    1
    .
    .
    .
```
Here, each line represents an edge which contains three tokens ```edge_type, head, tail, and weight ```.

## Running the code
To train OVRAU model on the example data, you can simply use the following command:
```
python src/main.py --input data 
```

The proposed model presents three possible variants depending on the used aggregator function. 
The user must also specify a --model, the variants of which are described in detail in the paper:

graphsage_mean -- GraphSage with mean-based aggregator
graphsage_seq -- GraphSage with LSTM-based aggregator
graphsage_maxpool -- GraphSage with max-pooling aggregator (as described in the NIPS 2017 paper)
graphsage_meanpool -- GraphSage with mean-pooling aggregator (a variant of the pooling aggregator, where the element-wie mean replaces the element-wise max).
