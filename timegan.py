import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from itertools import chain
from utils import batch_generator

class RNNnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation_fn=torch.sigmoid):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

class Loss:
    def __init__(self, params):
        self.params = params

    def loss_e_t0(self, x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def loss_e_0(self, loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def loss_e(self, loss_e_0, loss_s):
        return loss_e_0 + 0.1* loss_s

    def loss_s(self, h_hat_supervise, h):
        return F.mse_loss(h[:, 1:, :], h_hat_supervise[:, 1:, :])

    def loss_g_u(self, y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def loss_g_u_e(self, y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def loss_g_v(self, x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) - torch.sqrt(torch.var(x, 0) + 1e-6))
        )
        loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def loss_g(self, loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return loss_g_u + self.params.gamma * loss_g_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_g_v

    def loss_d(self, y_real, y_fake, y_fake_e):
        loss_d_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
        loss_d_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))
        return loss_d_real + loss_d_fake + self.params.gamma * loss_d_fake_e

def timegan(ori_data, params):
    embedder = RNNnet(params.input_size, params.hidden_size, params.hidden_size, params.num_layers).to(params.device)
    recovery = RNNnet(params.hidden_size, params.hidden_size, params.input_size, params.num_layers).to(params.device)
    generator = RNNnet(params.input_size, params.hidden_size, params.hidden_size, params.num_layers).to(params.device)
    supervisor = RNNnet(params.hidden_size, params.hidden_size, params.hidden_size, params.num_layers - 1).to(params.device)
    discriminator = RNNnet(params.hidden_size, params.hidden_size, 1, params.num_layers, activation_fn=None).to(params.device)
    
    #Losses
    loss = Loss(params)

    # Optimizers for the models, Adam optimizer
    optimizer_er = optim.Adam(chain(embedder.parameters(), recovery.parameters()))
    optimizer_gs = optim.Adam(chain(generator.parameters(), supervisor.parameters()))
    optimizer_d = optim.Adam(discriminator.parameters())
    
    embedder.train()
    generator.train()
    supervisor.train()
    recovery.train()
    discriminator.train()
    
    # Batch generator, it keeps on generating batches of data
    data_gen = batch_generator(ori_data, params)

    print("Start Embedding Network Training")
    for step in range(1, params.max_steps + 1):
        # Get the real batch data, and synthetic batch data. 
        x = data_gen.__next__() 
        h = embedder(x)
        x_tilde = recovery(h)
        loss_e_t0 = loss.loss_e_t0(x_tilde, x)
        loss_e_0 = loss.loss_e_0(loss_e_t0)
        optimizer_er.zero_grad()
        loss_e_0.backward()
        optimizer_er.step()

        if step % params.print_every == 0:
            print("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", loss_e: "+ str(np.round(np.sqrt(loss_e_t0.item()), 4)))
    print("Finish Embedding Network Training")

    print("Start Training with Supervised Loss Only")
    for step in range(1, params.max_steps + 1):
        # Get the real batch data, and synthetic batch data. 
        x = data_gen.__next__()
        z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)
        #print(z.shape)
        h = embedder(x)
        h_hat_supervise = supervisor(h)

        loss_s = loss.loss_s(h_hat_supervise, h)
        optimizer_gs.zero_grad()
        loss_s.backward()
        optimizer_gs.step()

        if step % params.print_every == 0:
            print("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", loss_s: "+ str(np.round(np.sqrt(loss_s.item()), 4)))
    print("Finish Training with Supervised Loss Only")

    print("Start Joint Training")
    for step in range(1, params.max_steps + 1):
        for _ in range(2):
            x = data_gen.__next__()
            z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)

            h = embedder(x)
            e_hat = generator(z)
            #print("e_hat: {}".format(e_hat.shape))
            h_hat = supervisor(e_hat)
            #print("h_hat: {}".format(h_hat.shape))
            h_hat_supervise = supervisor(h)
            x_hat = recovery(h_hat)
            #print("x_hat: {}".format(x_hat.shape))
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            loss_s = loss.loss_s(h_hat_supervise, h)
            loss_g_u = loss.loss_g_u(y_fake)
            loss_g_u_e = loss.loss_g_u_e(y_fake_e)
            loss_g_v = loss.loss_g_v(x_hat, x)
            loss_g = loss.loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v)
            optimizer_gs.zero_grad()
            loss_g.backward()
            optimizer_gs.step()

            h = embedder(x)
            x_tilde = recovery(h)
            h_hat_supervise = supervisor(h)

            loss_e_t0 = loss.loss_e_t0(x_tilde, x)
            loss_e_0 = loss.loss_e_0(loss_e_t0)
            loss_s = loss.loss_s(h_hat_supervise, h)
            loss_e = loss.loss_e(loss_e_0, loss_s)
            optimizer_er.zero_grad()
            loss_e.backward()
            optimizer_er.step()

        x = data_gen.__next__()
        z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)

        h = embedder(x)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        y_fake = discriminator(h_hat)
        y_real = discriminator(h)
        y_fake_e = discriminator(e_hat)

        loss_d = loss.loss_d(y_real, y_fake, y_fake_e)
        if loss_d.item() > 0.15:
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        if step % params.print_every == 0:
            print("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", loss_d: "+ str(np.round(loss_d.item(), 4))+ ", loss_g_u: "+ str(np.round(loss_g_u.item(), 4))+ ", loss_g_v: "+ str(np.round(loss_g_v.item(), 4))+ ", loss_s: "+ str(np.round(np.sqrt(loss_s.item()), 4))+ ", loss_e_t0: "+ str(np.round(np.sqrt(loss_e_t0.item()), 4)))
    print("Finish Joint Training")
    
    x = data_gen.__next__()
    z = torch.randn(ori_data.shape[0], x.size(1), x.size(2)).to(params.device)
    e_hat = generator(z)
    h_hat = supervisor(e_hat)
    h_hat_supervise = supervisor(h)
    x_hat = recovery(h_hat)
    
    synthetic_samples = x_hat.detach().cpu().numpy()
    return synthetic_samples
