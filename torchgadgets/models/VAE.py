import torch
import torch.nn as nn

from .NeuralNetwork import build_model


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_size: tuple(int),
                         encoder_layers: list(dict), 
                            decoder_layers: list(dict),
                                latent_dim: tuple(int)):
        super(VariationalAutoEncoder, self).__init__()

        self.input_size = input_size        
        self.encoder = build_model(encoder_layers)
        self.decoder = build_model(decoder_layers)

        self.fc_mu = nn.Linear(latent_dim[0], latent_dim[1])
        self.fc_sigma = nn.Linear(latent_dim[0], latent_dim[1])
    
    def forward(self, x):
        z = self.encode(x)
        x_out = self.decode(z)
        x_out = x_out.view(-1, *self.input_size)
        return x_out

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def encode(self, x):
        self.encoder(x)
        
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        z = self.reparametrize(mu, sigma)

        return z

    
    def decode(self, z):
        return self.decoder(z)

