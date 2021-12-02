# CRAAE
This is the implementation of CRAAE model for my paper "Calibrated Reconstruction Based Adversarial AutoEncoder Model for Novelty Detection". Some codes are based on GPND paper https://github.com/podgorskiy/GPND.

Experiments were implemented on the Clusters of a University.
Experiments enviroments settings are as below:

### Requirements

- Python 3.8.8  
- numpy>=1.8.2
- matplotlib>=1.5.3
- scipy>=1.1.0
- Pillow>=5.1.0
- scikit_learn>=0.19.1
- torch>=0.4.0
- torchvision>=0.2.1

- OS: Red Hat Enterprise Linux Server release 7.9

### File Explanation

* **partition_mnist.py, partition_cifar.py, partition_svhn.py, partition_tinyimg.py** - code for preparing datasets.
* **train_CRAAE.py** - code for training the CRAAE model on image datasets.
* **train_CRAAE_nw.py** - code for training the CRAAE model on network traffic dataset.
* **novelty_detector.py** - code for running novelty detector on image datasets.
* **novelty_detector_nw.py** - code for running novelty detector on network traffic dataset.
* **net.py** - contains definitions of network architectures for image datasets. 
* **net_nw_newmodel.py** - contains definitions of network architectures for network traffic dataset.
* **dataloading_out.py** - prepare traing and test batches for image datasets.
* **dataloading_nw.py** - prepare traing and test batches for network traffic dataset.
* **defaults.py** - set the hyperparameters and other default settings for image datasets.
* **defaults_nw.py** - set the hyperparameters and other default settings for network traffic dataset.

### Run steps

Step 1: run **partition_*.py** .

Step 2: run **CRAAE_pipeline.py** or **CRAAE_pipeline_nw.py**. Reusults will be written to **results.csv** file
