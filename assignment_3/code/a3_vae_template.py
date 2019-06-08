import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist
import numpy as np
from tensorboardX import SummaryWriter
from scipy.stats import norm


writer = SummaryWriter()


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        # input dim
        self.input_dim = 784
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim,z_dim)
        self.fc22 = nn.Linear(hidden_dim,z_dim)

        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, std = None, None
        
        x = self.fc1(input)
        x = self.relu(x)

        mean = self.fc21(x)
        std = self.softplus(self.fc22(x))

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        # output dim
        self.output_dim = 784
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.output_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = None
        x = self.relu(self.fc3(input))
        x = self.fc4(x)
        mean = self.sigmoid(x)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device="cpu"):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

        self.device = device

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None
        
        mean, std = self.encoder(input)
        
        # reparameterize
        eps = torch.randn_like(std).to(self.device)
        z = mean + (eps * std)

        x_hat = self.decoder(z)

        # recon loss
        loss_pxz = input*torch.log(x_hat+1e-8) + (1-input)*torch.log(1-x_hat+1e-8)
        loss_pxz = torch.sum(loss_pxz,dim=1)
        # reg loss
        kl_loss = 0.5*(1+torch.log(std**2) - std**2 - mean**2)
        kl_loss = torch.sum(kl_loss,dim=1)

        average_negative_elbo = -torch.mean(loss_pxz + kl_loss)

        return average_negative_elbo, x_hat

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None

        z = torch.normal(torch.zeros(n_samples,self.z_dim), torch.ones(n_samples,self.z_dim)).to(self.device)
        sampled_ims = self.decoder(z)

        im_means = torch.mean(sampled_ims, dim=0)


        return sampled_ims


def epoch_iter(model, data, optimizer, epoch=0, device="cpu"):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    average_epoch_elbo = None
    elbo_loss = 0
    step = 0

    data_length = len(data)
    updates = 0
    for batch_index, batch_inputs in enumerate(data):
        inputs = batch_inputs.view(batch_inputs.shape[0],-1)

        updates += 1
        if model.training:
            optimizer.zero_grad()

            elbo, out_images = model(inputs.to(device))
            elbo_loss += elbo.item()

            elbo.backward()
            optimizer.step()

            # step +=1
            # print("Step: {}/{} | Loss: {}".format(step,data_length, elbo))

        else:
            with torch.no_grad():
                elbo, _ = model(inputs.to(device))
                elbo_loss += elbo.item()


    average_epoch_elbo = elbo_loss/updates

    return -1*average_epoch_elbo


def run_epoch(model, data, optimizer, epoch=0, device="cpu"):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, epoch=epoch, device=device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device=device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    # random seed
    torch.manual_seed(42)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch % (int((ARGS.epochs-1)/2)) == 0:
            with torch.no_grad():
                n_samples=100
                sampled_imgs = model.sample(n_samples)
                sampled_imgs = sampled_imgs[:n_samples,:]

                imgs = make_grid(sampled_imgs.view(n_samples,1,28,28),nrow=10)
                imgs = imgs.cpu().numpy()
                im_plot = plt.imshow(np.transpose(imgs,(1,2,0)),cmap='gray')
                plt.savefig('img_vae_gen_'+str(epoch)+'.png')


        elbos = run_epoch(model, data, optimizer, device=device, epoch=epoch)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        writer.add_scalar('ELBO/Train', train_elbo, epoch)
        writer.add_scalar('ELBO/Test', val_elbo, epoch)


    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim==2:
        model.eval()
        with torch.no_grad():
            num_points=20
            digit_size = 28
            manifold = np.zeros((digit_size*num_points, digit_size *num_points))

            xx = norm.ppf(np.linspace(0.05,0.95,num_points))
            yy = norm.ppf(np.linspace(0.05,0.95,num_points))

            for i, yi in enumerate(yy):
                for j,xi in enumerate(xx):
                    z = torch.from_numpy((np.array([[xi, yi]]))).to(device)
                    x_hat = model.decoder(z.float())
                    x = x_hat.view(digit_size, digit_size)
                    manifold[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = x

            plt.figure(figsize=(10,10))
            plt.imshow(manifold, cmap='gray')
            plt.savefig("manifold.png")
    


    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--n_samples', default=100, type=int,
                    help='number of samples to plot during training')
    

    ARGS = parser.parse_args()

    main()
