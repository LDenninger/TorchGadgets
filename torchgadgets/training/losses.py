import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconKLDivLoss(nn.Module):
    """
        Combined loss function of the MSE reconstruction loss and the KL divergence.
    
    """

    def __init__(self, lambda_kld=1e-3):
        super(ReconKLDivLoss, self).__init__()
        self.lambda_kld = lambda_kld

    def forward(self, output, target, mu, log_var):
        recons_loss = F.mse_loss(output, target)
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
        loss = recons_loss + self.lambda_kld * kld

        return loss, (recons_loss, kld)


