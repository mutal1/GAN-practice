import torch 
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

#nosie vectior def
latent_dim = 100

#Generator class def
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #block def
        def block(input_dim, output_dim , normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                #batch normalize 
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128 , normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024, 1*28*28),
            nn.Tanh()
        )

    def forward(self,z):
        img = self.model(z)
        img = img.view(img.size(0),1,28,28)

        return img
    
    #Discriminator class def
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28 , 512),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    def forward(self,img):
        flattened = img.view(img.size(0),-1)
        output = self.model(flattened)

        return output
    
transforms_train = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./dataset' , train = True, download = True , transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 128 , shuffle = True , num_workers = 4) # num_workers data speed

generator = Generator()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()

#loss function
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

#learning rate def
lr = 0.0002

#optimize
optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr , betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr , betas= (0.5,0.999))

import time 

n_epochs = 200
sample_interval = 2000 # batch ㅁㅏㄷㅏ output 
start_time = time.time()

for epoch in range(n_epochs):
    for i , (imgs,_) in enumerate(dataloader):

        #real and fake image label create
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) 
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0)

        real_imgs = imgs.cuda()

        #generator train
        optimizer_G.zero_grad()

        #random noize sampling
        z = torch.normal(mean=0, std = 1 , size= (imgs.shape[0], latent_dim)).cuda()

        #image create
        generated_imgs = generator(z)

        #generator loss value cal
        g_loss = adversarial_loss(discriminator(generated_imgs), real)

        #generator parameter update
        g_loss.backward()
        optimizer_G.step()

        #discriminator train
        optimizer_D.zero_grad()

        # discriminator loss cal
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        #discriminator parameter update
        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i 
        if done % sample_interval == 0 : 
            # create image output 
            save_image(generated_imgs.data[:25], f'{done}.png', nrow=5 , normalize = True)
        
        #log output
        print(f'[Epoch {epoch} / {n_epochs}] [D loss:{d_loss.item():.6f}] [G loss:{g_loss.item():.6f}] [Elapsed time:{time.time()-start_time:.2f}]')

        from IPython.display import Image

        Image('92000.png')


