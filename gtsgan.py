from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from itertools import chain
from utils import batch_generator
from models.models import Embedder, Recovery, Generator, Supervisor, Discriminator, FusionBlock
from models.gae import GraphEncoder

class Loss:
    def __init__(self, params):
        self.params = params
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def E_loss_T0(self, x_tilde, x):
        #Embedder Network loss
        return self.mse(x_tilde, x)
    
    def E_loss0(self, E_loss_T0):
        #Embedder Network loss
        return 10*torch.sqrt(E_loss_T0)

    def E_loss(self, E_loss0, G_loss_S):
        #Embedder Network loss
        return E_loss0 + 0.1*G_loss_S
    
    def supervised_loss_graph(self, h, h_hat_supervise):
        # Supervised loss for graph embedding
        return self.mse(h[:, :, 1:], h_hat_supervise[:,:,:-1])

    def G_loss_S(self, h, h_hat_supervise):
        # Supervised loss
        return self.mse(h[:, 1:, :], h_hat_supervise[:,:-1,:])

    # Generator Losses
    def G_loss_U(self, y_fake):
        # Adversarial loss
        return self.bce(y_fake, torch.ones_like(y_fake))

    def G_loss_U_e(self, y_fake_e):
        # Adversarial loss
        return self.bce(y_fake_e, torch.ones_like(y_fake_e))

    def G_loss_V(self, x_hat, x):
        # Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) - torch.sqrt(torch.var(x, 0) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return G_loss_V1 + G_loss_V2

    def G_loss(self, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V):
        # Summation of G loss
        return G_loss_U + self.params.gamma*G_loss_U_e + 100*torch.sqrt(G_loss_S) + 100*G_loss_V

    def D_loss(self, y_real, y_fake, y_fake_e): 
        # Discriminator loss
        D_loss_real = self.bce(y_real, torch.ones_like(y_real))
        D_loss_fake = self.bce(y_fake, torch.zeros_like(y_fake))
        D_loss_fake_e = self.bce(y_fake_e, torch.zeros_like(y_fake_e))
        return D_loss_real + D_loss_fake + self.params.gamma*D_loss_fake_e

def gtsgan(ori_data, params):
    encoder = Embedder(params.input_size, params.hidden_size, params.num_layers).to(params.device)
    structural_encoder = GraphEncoder(params, node_num= params.num_nodes, dim = params.graph_hidden, input_dim = params.graph_input, topk = params.top_k).to(params.device)
    fusion = FusionBlock(params).to(params.device)
    decoder = Recovery(params.hidden_size, params.input_size, params.num_layers).to(params.device)
    generator = Generator(params.input_size, params.hidden_size, params.num_layers).to(params.device)
    supervisor = Supervisor(params.hidden_size, params.num_layers - 1).to(params.device)
    discriminator = Discriminator(params.hidden_size, params.num_layers).to(params.device)
    
    #Losses
    loss = Loss(params)

    # Optimizers for the models, Adam optimizer
    optimizer_encoder = optim.Adam(encoder.parameters())
    optimizer_structural_encoder = optim.Adam(structural_encoder.parameters())
    optimizer_decoder = optim.Adam(decoder.parameters())
    optimizer_fusion = optim.Adam(fusion.parameters())
    optimizer_generator = optim.Adam(generator.parameters())
    optimizer_supervisor = optim.Adam(supervisor.parameters())
    optimizer_discriminator = optim.Adam(discriminator.parameters())

    
    encoder.train()
    structural_encoder.train()
    fusion.train()
    decoder.train()
    generator.train()
    supervisor.train()
    discriminator.train()
    
    # Batch generator, it keeps on generating batches of data
    data_gen = batch_generator(ori_data, params)

    print("Start Embedding Network Training with reconstruction loss")
    for step in range(params.max_steps):
        # Get the real batch data, and synthetic batch data. 
        x = data_gen.__next__() 
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        h = encoder(x)
        x_tilde = decoder(h)
        E_loss_T0 = loss.E_loss_T0(x_tilde, x)
        E_loss0 = loss.E_loss0(E_loss_T0)
        E_loss0.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        if step % params.print_every == 0:
            print("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", e_loss: "+ str(np.round(np.sqrt(E_loss_T0.item()), 4)))
            
    print("Finish Embedding Network Training with reconstruction loss")

    print("Start Embedding Network Training with supervision loss")
    for step in range(params.max_steps):
        # Get the real batch data, and synthetic batch data. 
        x = data_gen.__next__() 
        optimizer_supervisor.zero_grad()
        h = encoder(x)
        h_hat_supervise = supervisor(h)
        G_loss_S = loss.G_loss_S(h, h_hat_supervise)
        G_loss_S.backward()
        optimizer_supervisor.step()

        if step % params.print_every == 0:
            print("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", s_loss: "+ str(np.round(np.sqrt(G_loss_S.item()), 4)))



    print("Start Joint Training")
    for step in range(params.max_steps):
        for _ in range(2):
            # Train the Generator
            optimizer_generator.zero_grad()
            x = data_gen.__next__()
            z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)

            h1 = encoder(x)
            h2, learned_graph, gated_edge_index= structural_encoder(x)

            h = fusion(h1, h2)

            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            h_hat_supervise = supervisor(h)
            x_hat = decoder(h_hat)

            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            G_loss_U = loss.G_loss_U(y_fake)
            G_loss_U_e = loss.G_loss_U_e(y_fake_e)
            G_loss_S = loss.G_loss_S(h, h_hat_supervise)
            G_loss_V = loss.G_loss_V(x_hat, x)

            G_loss = loss.G_loss(G_loss_U, G_loss_U_e, G_loss_S, G_loss_V)
            G_loss.backward()
            optimizer_generator.step()

            # Train the  Fusion Network
            optimizer_supervisor.zero_grad()
            optimizer_fusion.zero_grad()
            optimizer_structural_encoder.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            h1 = encoder(x)
            h2, learned_graph, gated_edge_index = structural_encoder(x)
            h = fusion(h1, h2)
            
            x_tilde = decoder(h)
            h_hat_supervise = supervisor(h)

            E_loss_T0 = loss.E_loss_T0(x_tilde, x)
            E_loss0 = loss.E_loss0(E_loss_T0)
            G_loss_S = loss.G_loss_S(h, h_hat_supervise)
            E_loss = loss.E_loss(E_loss0, G_loss_S)
            
            E_loss.backward()
            optimizer_supervisor.step()
            optimizer_fusion.step()
            optimizer_structural_encoder.step()
            optimizer_encoder.step()
            optimizer_decoder.step()

        # Discriminator training 
        x = data_gen.__next__()
        z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)
        
        h1 = encoder(x)
        h2, learned_graph, gated_edge_index = structural_encoder(x)
        h = fusion(h1, h2)

        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        
        y_real = discriminator(h)
        y_fake = discriminator(h_hat)
        y_fake_e = discriminator(e_hat)

        loss_d = loss.D_loss(y_real, y_fake, y_fake_e)

        # Train discriminator (only when the discriminator does not work well)
        if loss_d.item() > 0.15:
            optimizer_discriminator.zero_grad()
            loss_d.backward()
            optimizer_discriminator.step()

        if step % params.print_every == 0:
            print("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", d_loss: "+ str(np.round(loss_d.item(), 4))+ ", g_loss_u: "+ str(np.round(G_loss_U.item(), 4))+  ", g_loss_s: "+ str(np.round(np.sqrt(G_loss_S.item()), 4))+ ", g_loss_v: "+ str(np.round(G_loss_V.item(), 4))+ ", e_loss_t0: "+ str(np.round(np.sqrt(E_loss_T0.item()), 4)))
    print("Finish Joint Training")

    
    with torch.no_grad():
        x = data_gen.__next__()
        z = torch.randn(ori_data.shape[0], x.size(1), x.size(2)).to(params.device)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        x_hat = decoder(h_hat)

        synthetic_samples = x_hat.detach().cpu().numpy()
        new_edge_index = gated_edge_index.detach().cpu().numpy()
        learned_graph = learned_graph.detach().cpu().numpy()
        return synthetic_samples, new_edge_index, learned_graph