# TorchGadgets
This project aims at improving the workflow when working with PyTorch models. It capsules typically redundant code that is reused often when training your models.
The overall goal is to define everything about your model in a standardized configuration file. The complete handling of the initialization, training, logging and visualization can ultimately be done by through the package byu only passing the configuration defined in a dictionary.

## Installation
In your python environment: `python setup.py install`

## Structure
    
    TorchGadgets            # TorchGadgets Package
        ├── data            # Data loading and processing
        ├── evaluation      # Implementation of different evaluation metrics and complete evaluatioan scripts
        ├── logging         # Logging of metrics and training progress   
        ├── models          # Neural Network template class and further PyTorch models
        ├── tools           # Useful helper functions and debugging modules
        ├── training        # Training scripts and custom scheduler modules

## Usage

TODO: Describe all modules

The main idea is that we can define config dictionaries for each module of the training pipeline (model, optimizer, scheduler, data augmentation etc.) and leave everything else to the package.
For further elaboration on the configuration data structres, please look into the section "Configuration Structures". Most functions typically just take the configuration as a parameter and do not require any external definition of modules. If however one wants to use different modules than defined through the config, it is typically possible to also pass the module as a parameter preventing TorchGadgets from initializing the module from the config.

### Model Training
Train a model: `logger = torchgadgets.training.trainNN(config=MODEL_CONFIG)`
 - The returned logger module saved all information from the training and can be used to visualize or compare the training of the model

### Inspector Gadgets
This is the debugging tool of the package using the IPDB debugger. It can be used to set breakpoints in the forwartd and backward pass or to log weights and activations.

## Configuration Structures

TODO: Provide complete documentation on all options.

#### Architecture Configuration
The architecture config is basically a list of sequential modules where each module config is defined by the dictionary.

Example config:

    MODEL_ARCHITECTURE = [
        {'type': 'ResNet', 'size': 18, 'remove_layer': 1, 'weights': 'DEFAULT'},
        {'type': 'flatten'},
        {'type': 'dropout', 'prob': 0.3},
        {'type': 'linear', 'in_features': tg.models.RESNET_FEATURE_DIM[18][1], 'out_features': 2}
    ]


#### Scheduler Configuration

Example config:
    
    SCHEDULER = {'epoch_scheduler': {'type': 'exponential', 'gamma': 0.9}, 'iteration_scheduler': None}
    
    

#### Optimizer Configuration

Example config:
    
    OPTIMIZER = {'type': 'Adam'}
    
    

#### Data Augmentation Configuration
The data augmentation config is a list that defines each processing step of our data augmentation pipeline.

Example config:
    
    DATA_AUGMENTATION = [   
        {'type': 'mixup', 'alpha': 1.0, 'prob': 0.5, 'beta': 1.0, 'num_classes': 2, 'train': True, 'eval': False},
        {'type': 'random_rotation','degrees': 5, 'train': True, 'eval': False},
        {'type': 'random_horizontal_flip','prob': 0.3, 'train': True, 'eval': False},
        {'type': 'color_jitter', 'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.3, 'hue': 0.0, 'train': True, 'eval': False},
        {'type': 'gaussian_blur', 'kernel_size':(5,5), 'sigma': (0.1,2.0), 'train': True, 'eval': False},
        {'type': 'normalize', 'train': True, 'eval': True},
    ]
    

#### Model Configuration
This configuration basically now bundles all previously configuration to the configuration of a single model or training run.

Example config:
    
    MODEL_CONFIG = {
        'layers': MODEL_ARCHITECTURE,
        'output_dim': 2,
        'activation': 'relu',
        'batch_size': 32,
        'num_epochs': 10,
        'num_iterations': 0,
        'learning_rate': 0.00003,
        'random_seed': 22,
        'pre_processing': DATA_AUGMENTATION,
        'evaluation': {
            'metrics': ['accuracy'],
            'frequency': 1
        },
        'dataset': {
                'name': 'oxfordpets',
                'train_size': 2000,
                'val_size': 0,
                'test_size': 3669,
                'train_shuffle': True,
                'eval_shuffle': False,
                'drop_last': True,
                'classes': [0, 1]
        },
        'scheduler': SCHEDULER,
        'optimizer': OPTIMIZER
                        
    }