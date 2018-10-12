import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.linear0 = nn.Linear(args.latent_dim,128)
        self.linear1 = nn.Linear(128, 256)
        self.bnorm1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 512)
        self.bnorm2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1024)
        self.bnorm3 = nn.BatchNorm1d(1024)
        self.linear4 = nn.Linear(1024, 784)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Generate images from z
        x = self.linear0(z)
        x = self.LeakyReLU(x)
        x = self.linear1(x)
        x = self.bnorm1(x)
        x = self.LeakyReLU(x)
        x = self.linear2(x)
        x = self.bnorm2(x)
        x = self.LeakyReLU(x)
        x = self.linear3(x)
        x = self.bnorm3(x)
        x = self.LeakyReLU(x)
        x = self.linear4(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.input_dim = input_dim

        self.linear0 = nn.Linear(self.input_dim, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 1)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()


    def forward(self, img):
        # return discriminator score for img

        x = self.linear0(img)
        x = self.LeakyReLU(x)
        x = self.linear1(x)
        x = self.LeakyReLU(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        return x.squeeze()


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device="cuda:0"):

    criterion = nn.BCELoss()

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            imgs = imgs.view(batch_size,-1).to(device)

            true_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            z = torch.randn((batch_size, args.latent_dim)).to(device)
            fake_imgs = generator(z)
            pred_z = discriminator(fake_imgs)

            loss_gen = criterion(pred_z, true_labels)
            
            loss_gen.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            pred_x = discriminator(imgs)

            loss_true = criterion(pred_x, true_labels)

            # z_new = torch.randn((batch_size, args.latent_dim)).to(device)
            # fake_imgs_new = generator(z_new)
            # pred_z_new = discriminator(fake_imgs_new)
            # loss_gen = criterion(pred_z_new, fake_labels)

            loss_fake = criterion(pred_z.detach(), fake_labels)

            loss_dm = 0.5*(loss_true + loss_fake)

            loss_dm.backward()
            optimizer_D.step()
            

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                gen_imgs = fake_imgs.view(args.batch_size,1,28,28)
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

                print("Epoch: {}| step: {}/{}| Loss:: Gen: {}| Dis: {}".format(epoch,i,len(dataloader),loss_gen.item(), loss_dm.item()))
                


def main():
    # Random seed
    torch.manual_seed(42)

    #Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    input_dim = 784
    generator = Generator()
    discriminator = Discriminator(input_dim)

    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device=device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
