import torch
from tqdm import tqdm

from .classification_metrics import evaluate

@torch.no_grad()
def run_evaluation( model: torch.nn.Module, 
                    data_augmentor,
                    dataset: torch.utils.data.DataLoader, 
                    config: dict,
                    evaluation_metrics = None,
                    criterion = None,
                    suppress_output: bool = False):
    """
        Runs evaluation of the given model on the given dataset.

        Arguments:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            config (dict): The configuration dictionary.
            device (str, optional): The device to evaluate on. Defaults to "cpu".

    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_iterations = config['num_eval_iterations'] if config['num_eval_iterations'] != -1 else len(dataset)
    import ipdb; ipdb.set_trace()
    
    # Setup model for evaluation
    model.eval()
    model.to(device)
    eval_metrics = {}
    if suppress_output:
        progress_bar = enumerate(dataset)
    else:
        progress_bar = tqdm(enumerate(dataset), total=num_iterations)
        progress_bar.set_description(f'Evaluation:')
    outputs = []
    targets = []
    losses = []
    for i, (imgs, labels) in progress_bar:
        if i==num_iterations:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        # apply preprocessing surch as flattening the imgs and create a one hot encodinh of the labels
        imgs, labels = data_augmentor((imgs, labels), train=False)
        output = model(imgs)
        outputs.append(output.cpu())
        targets.append(labels.cpu())
        if criterion is not None:
            loss = criterion(output, labels.float())
            losses.append(loss.cpu().item())

    eval_metrics = evaluate(torch.stack(outputs, dim=0), torch.stack(targets, dim=0), config=config, metrics=evaluation_metrics)

    if criterion is None:
        return eval_metrics
    
    return eval_metrics, sum(losses) / len(losses)

@torch.no_grad()
def run_vae_evaluation( model: torch.nn.Module, 
                    data_augmentor,
                    dataset: torch.utils.data.DataLoader, 
                    config: dict,
                    evaluation_metrics = None,
                    criterion = None,
                    suppress_output: bool = False):
    """
        Runs evaluation of the given model on the given dataset.

        Arguments:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            config (dict): The configuration dictionary.
            device (str, optional): The device to evaluate on. Defaults to "cpu".

    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_iterations = config['num_eval_iterations'] if config['num_eval_iterations'] != -1 else len(dataset)
    
    # Setup model for evaluation
    model.eval()
    model.to(device)
    eval_metrics = {}
    if suppress_output:
        progress_bar = enumerate(dataset)
    else:
        progress_bar = tqdm(enumerate(dataset), total=num_iterations)
        progress_bar.set_description(f'Evaluation:')
    outputs = []
    targets = []
    losses = []
    mse_list = []
    kld_list = []
    for i, (imgs, labels) in progress_bar:
        if i==num_iterations:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        # apply preprocessing surch as flattening the imgs and create a one hot encodinh of the labels
        img_augm, labels_augm = data_augmentor((imgs, labels), train=False)
        output, (z, mu, sigma) = model(img_augm)
            # Compute loss
        outputs.append(output.cpu())
        targets.append(labels.cpu())
        if criterion is not None:
            loss, (mse, kld) = criterion(output.float(), img_augm.float(), mu, sigma)
            losses.append(loss.cpu().item())
            mse_list.append(mse.cpu().item())
            kld_list.append(kld.cpu().item())

    eval_metrics = evaluate(torch.stack(outputs, dim=0), torch.stack(targets, dim=0), config=config, metrics=evaluation_metrics)

    if criterion is None:
        return eval_metrics
    
    return eval_metrics, sum(losses) / len(losses), sum(mse_list) / len(mse_list), sum(kld_list) / len(kld_list), 

