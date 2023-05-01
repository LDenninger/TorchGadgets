import torch
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self,
                exp_name: str = None,
                    run_name: str = None,
                        base_dir: str = None,
                            model_config: dict = None,
                                writer: SummaryWriter = None,
                                    save_external: bool = True,                                    
                                    save_internal: bool = False, 
                                ):
        
        ### Run Information ###
        
        self.exp_name = exp_name
        self.run_name = run_name

        if save_external:
            assert base_dir is not None and exp_name is not None and run_name is not None, 'Please provide exp.-, run name and base_dir.'
            self.writer = SummaryWriter(str(self.run_dir/"logs")) if writer is None else writer

        self.model_config = model_config

        self.save_internal = save_internal
        self.save_external = save_external
        self._internal_log = {}

        self.log_gradients = False

        
        
    def log_data(self, epoch: int, data: dict,  iteration: int=None):
        """
            Log the data.
            Arguments:
                epoch (int): The current epoch. 
                data (dict): The data to log.
                    Format:
                            {
                                [name]: value,
                                ...
                            }
        
        """
        if self.save_external:
            
            for key, value in data.items():
                if type(value) == list:
                    prefix_name = f'iteration_metrics/'
                    log_iter = self.model_config['num_iterations']*(epoch-1)
                    for i, item in enumerate(value):
                        self.writer.add_scalar(prefix_name + key, item, log_iter + i +1)
                    continue
                prefix_name = f'epoch_metrics/' if iteration is None else f'iteration_metrics/'
                log_iter = epoch if iteration is None else (self.model_config['num_iterations']*(epoch-1) + iteration)
                self.writer.add_scalar(prefix_name + key, value, log_iter)
            
        if self.save_internal:
            self._save_internal(data)
    
    def get_log(self):
        return self._internal_log

    def get_last_log(self):
        last_log = {}
        for key in self._internal_log.keys():
            last_log[key] = self._internal_log[key][-1]
        return last_log

    def enable_internal_log(self):
        self.save_internal = True
    
    def disable_internal_log(self):
        self.save_internal = False

    
    def _save_internal(self, data):
        for key, value in data.items():
            if not key in self._internal_log.keys():
                self._internal_log[key] = []
            if type(value) == list:
                self._internal_log[key] += value
                continue
            self._internal_log[key].append(value)