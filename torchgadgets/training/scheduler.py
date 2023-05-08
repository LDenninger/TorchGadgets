import numpy as np
import torch

###--- Custom Scheduler Class ---###
# Right now it only includes a custom scheduler
# It should eventually be a Interface for all other PyTorch schedulers 
# such that custom and standard schedulers can be mixed arbitrarly with the training functions

class Scheduler:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.scheduler_config = config
        self._init_scheduler()
    
    def step(self, step: int):
        learning_rate = self.schedule_step(step)
        self.set_learning_rate(learning_rate)


    def set_learning_rate(self, learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate
        
    def _init_scheduler(self):
        if self.scheduler_config['type']=='warmup_cosine_decay':
            self.schedule_step = self.warmup_cosine_decay
            self.warmup_steps = self.scheduler_config['warmup_steps']
            self.base_lr = self.scheduler_config['learning_rate']
            self.num_iterations = self.scheduler_config['num_iterations']

    def warmup_cosine_decay(self, step):
        """
            Custom learning rate scheduler. In the first warmup_steps steps the learning rate is linearly increased.
            After this points is reached the learning rate cosine decays.
        
        """
        warmup_factor = min(step, self.warmup_steps) / self.warmup_steps
        decay_step = max(step - self.warmup_steps, 0) / (self.num_iterations- self.warmup_steps)
        return self.base_lr * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2