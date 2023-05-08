# Torch Gadgets
The project aims to improve the workflow when working with PyTorch projects. It provides a base environment to run, log, debug and visualize PyTorch models.
The goal is to define everything needed for your PyTorch model in a standardized configuration dictionary. The rest is then handled by the training functions.
The provided logger can save all information from the training, in RAM or on the disk using TensorBoard, and save checkpoints of your model during training.
This enables an easy implementation into an existing experiment directory for logging. 