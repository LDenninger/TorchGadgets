import torch

###--- Initialization ---###
def initialize_optimizer(model: torch.nn.Module, config: dict):
    """
        Initialize optimizer.

        Arguments:
            model (torch.nn.Module): Model to be optimized.
            config (dict): Optimizer configuration dictionary.
          
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
    return optimizer

def initialize_loss( config: dict):
    """
        Initialize criterion.

        Arguments:
            config (dict): Criterion configuration dictionary.
                Format:
                     
    """
    criterion = torch.nn.CrossEntropyLoss()

    return criterion
