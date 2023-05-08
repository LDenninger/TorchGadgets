import numpy as np
import torch

###--- Custom Scheduler Class ---###
# Right now it only includes a custom scheduler
# It should eventually be a Interface for all other PyTorch schedulers 
# such that custom and standard schedulers can be mixed arbitrarly with the training functions

class SchedulerManager:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.scheduler_config = config['scheduler']

        self.epoch_scheduler = None
        self.iteration_scheduler = None
        self.warmup_scheduler = None

        self._num_iterations = config['num_iterations']
        self._num_epoch = config['num_epochs']
        self._base_lr = config['learning_rate']
        self._epoch = 0
        self._iteration = 0
        self._init_scheduler()
    
    def step(self, step: int):
        if step == 1:
            if self.epoch_scheduler is not None:
                self.epoch_scheduler.step()
            self._epoch += 1
        self._iteration = step
        if self.iteration_scheduler is not None:
            self.iteration_scheduler.step()

    #def _init_scheduler(self):


        #if self.scheduler_config['epoch_scheduler']['type']=='warmup_cosine_decay':
        #    self.epoch_scheduler = WarmUpCosineDecayLR(self.optimizer, self.scheduler_config['warmup


class  WarmUpCosineDecayLR():
    def __init__(self, optimizer, warmup_steps, num_steps, base_lr) -> None:
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.base_lr = base_lr
        self._step = 0

    def step(self):
        self._step += 1
        learning_rate = self._warmup_cosine_decay(self._step)
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def set_base_lr(self, base_lr):
        self.base_lr = base_lr

    def _warmup_cosine_decay(self, step):
        """
            Custom learning rate scheduler. In the first warmup_steps steps the learning rate is linearly increased.
            After this points is reached the learning rate cosine decays.
        
        """
        warmup_factor = min(step, self.warmup_steps) / self.warmup_steps
        decay_step = max(step - self.warmup_steps, 0) / (self.num_steps- self.warmup_steps)
        return self.base_lr * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2