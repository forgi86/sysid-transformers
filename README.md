# dynoGPT: model-free meta learning of dynamical systems with Transformers

This repository contains the Python code to reproduce the results of the paper dynoGPT: model-free meta learning of dynamical systems with Transformers (in preparation, 2023)
by Marco Forgione, Filippo Pura and Dario Piga.


* We introduce the concept of model-free meta learning for System Identification. 


# Block Diagram

The block diagram below illustrates the proposed multi-step simulation error minimization approach applied to a
state-space model. Quantities in red are tunable optimization variable (so as the parameters of the state and output
neural network mappings).
 
At each iteration of the gradient-based optimization loop:

1. A batch consisting of q length-m subsequences of measured input, measured output, and hidden state is extracted from the training 
dataset (and from the tunable hidden state sequence)
1. The system's simulated state and output subsequences are obtained by applying m-step-ahead simulation
 to the input subsequences. The initial condition is taken as the first element of the hidden state sequence 
1. The fit loss is computed as the discrepancy between measured and simulated output; the consistency 
  loss is computed as the discrepancy between hidden and simulated state; the total loss is a defined as a weighted
  sum of the fit and consistency loss
1. Derivatives of the total loss w.r.t. the hidden state and the neural network parameters are computed via
  back-propagation
1. Using the derivatives computed at the previous step, a gradient-based optimization step is performed. The hidden state and neural network parameters are updated 
  in the negative gradient direction, aiming to minimize the total loss

![GPT](fig/.decoder_architecture.pdf "Generalized one-step-ahead predictor")

# Software requirements:
Experiments were performed on a Python 3.11 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pytorch (>= 2.0)
 
These dependencies may be installed through the commands:

```
conda install numpy scipy pandas matplotlib
conda install pytorch -c pytorch
```

To run the software, please make sure that this repository's root folder is added to 
your PYTHONPATH.

# Citing

If you find this project useful, we encourage you to:

* Star this repository :star: 

* Cite the [paper](https://arxiv.org/abs/2206.12928) 
```
@article{forgione2023a,
  title={dynoGPT: model-free meta learning of dynamical systems with Transformers?},
  author={Forgione, M. and Pura, F. and Piga, D.},
  journal={arXiv preprint arXiv:2206.12928},
  year={2022}
}
```
