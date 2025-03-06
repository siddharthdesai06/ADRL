import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scipy import linalg
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.metrics import f1_score
from torchvision import models
import shutil
import pandas as pd
data_folder_butterfly = '/home/siddharth/siddharth/ADRL/Data/Butterfly'
data_folder_aug_butterfly = '/home/siddharth/siddharth/ADRL/Data/Butterfly_aug/Train_Aug'
data_folder_animals = '/home/siddharth/siddharth/ADRL/Data/animals'
data_folder_aug_animals = '/home/siddharth/siddharth/ADRL/Data/Animals_aug'


class AnimalsDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
        self.data = []
        self.label_map = {}
        for i, folder in enumerate(self.subfolders):
            for filename in os.listdir(folder):
                image_path = os.path.join(folder, filename)
                label_name = folder.split('/')[-1]
                self.label_map[label_name] = i
                self.data.append((image_path,i))          
        random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def transform(self, img):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return img_transform(img)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):  
            img = cv2.imread(self.data[idx][0])  
            label = self.data[idx][1] 
            img = self.transform(img)  
            return img, label
        
        elif isinstance(idx, slice):  
            indices = range(*idx.indices(len(self.data)))
            batch_imgs = []
            batch_labels = []
            for i in indices:
                img = cv2.imread(self.data[i][0])  
                img = self.transform(img)  
                batch_imgs.append(img)
                batch_labels.append(self.data[i][1])
            return torch.stack(batch_imgs), batch_labels

        elif isinstance(idx, list):  
            batch_imgs = []
            batch_labels = []
            for i in idx:
                img = cv2.imread(self.data[i][0])  
                img = self.transform(img)  
                batch_imgs.append(img)
                batch_labels.append(self.data[i][0])
            
            # Return batch of images and labels
            return torch.stack(batch_imgs), batch_labels
        
#train_dataset = AnimalsDataset(data_folder_aug_animals) 
class ButterflyDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.desc_path = f'{data_folder}/descriptions.csv'
        self.subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
        desc_df = pd.read_csv(self.desc_path)
        self.data = []
        self.label_map = {}
        count_labels = 0
        for filename in os.listdir(self.subfolders[0]):
            image_path = os.path.join(self.subfolders[0], filename)
            #print(image_path)
            image_name = filename.split('_flipped')[0]
            if(not image_name.endswith('.jpg')):
                image_name = image_name + '.jpg'
            label_name = desc_df[desc_df['filename'] == image_name]['label'].values[0]
            if label_name not in self.label_map:
                self.label_map[label_name] = count_labels
                count_labels += 1
            self.data.append((image_path, self.label_map[label_name]))
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def transform(self, img):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return img_transform(img)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):  
            #print(self.data[idx][0])
            img = cv2.imread(self.data[idx][0]) 
            #print(img) 
            label = self.data[idx][1] 
            img = self.transform(img)  
            return img, label
        
        elif isinstance(idx, slice):  
            indices = range(*idx.indices(len(self.data)))
            batch_imgs = []
            batch_labels = []
            for i in indices:
                img = cv2.imread(self.data[i][0])  
                img = self.transform(img)  
                batch_imgs.append(img)
                batch_labels.append(self.data[i][1])
            return torch.stack(batch_imgs), batch_labels

        elif isinstance(idx, list):  
            batch_imgs = []
            batch_labels = []
            for i in idx:
                img = cv2.imread(self.data[i][0])  
                img = self.transform(img)  
                batch_imgs.append(img)
                batch_labels.append(self.data[i][0])
            
            # Return batch of images and labels
            return torch.stack(batch_imgs), batch_labels


class VAE(nn.Module):
    def __init__(self, z_dim, layer_count=3, channels=3, d=128, mul=1):
        super(VAE, self).__init__()

        
        self.d = d
        self.z_dim = z_dim

        self.layer_count = layer_count

        self.mul = mul
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, self.d * self.mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * self.mul))
            inputs = self.d * self.mul
            self.mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 4 * 4, self.z_dim)
        self.fc2 = nn.Linear(inputs * 4 * 4, self.z_dim)

        self.d1 = nn.Linear(self.z_dim, inputs * 4 * 4)

        self.mul = inputs // self.d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d * self.mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(self.d *self. mul))
            inputs = self.d * self.mul
            self.mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

    def encode(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

        x = x.view(x.shape[0], self.d_max * 4 * 4)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.z_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 4, 4)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.z_dim, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
        
        
def loss_function(recon_x, x, mu, logvar,scaling_factor):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    
    return 0.1*BCE, scaling_factor * KLD

# Function to calculate the FID between two sets of images
def calculate_fid(real_images, generated_images, batch_size=32, device='cuda'):
    
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = nn.Identity()  
    model.eval()  
    act1 = get_activations(real_images, model, batch_size, device)
    act2 = get_activations(generated_images, model, batch_size, device)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score

def get_activations(images, model, batch_size, device):
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)
    activations = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            activations.append(pred.cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    return activations

# Function to preprocess images (resize and normalize)
def preprocess_images(images, image_size=(299, 299)):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x: (x + 1) / 2), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_images = [preprocess(img) for img in images]
    return torch.stack(processed_images)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid

def compute_fid(model,data_loc, z_dim):
    test_dataset = ButterflyDataset(data_loc)
    
    #Sample 1000 images from the dataset
    sample_indices = random.sample(range(len(test_dataset)), 1000)
    sample_images = [test_dataset[i][0] for i in sample_indices]
    model = model.to("cuda")
    #Generating 1000 images using the generator
    with torch.no_grad():
        z = torch.randn(1000, z_dim)
        z = z.to("cuda")
        img = model.decode(z.view(1000, z_dim, 1, 1))
        img = img.to("cuda")
        
    img_list = [img[i] for i in range(1000)]
    img_list[0].shape
        
    preprocessed_real_images = preprocess_images(sample_images)
    preprocessed_generated_images = preprocess_images(img_list)
    
    fid_score = calculate_fid(preprocessed_real_images, preprocessed_generated_images)
    return fid_score


def train(**kwargs): 
    z_dim = kwargs.get('z_dim', 128)
    batch_size = kwargs.get('batch_size', 128)
    train_epoch = kwargs.get('train_epoch', 40)
    model_path = kwargs.get('pretrained_model_path', None)
    lr = kwargs.get('lr', 0.0005)
    layer_count = kwargs.get('layer_count', 5)
    device = kwargs.get('device', 'cuda')
    train_data = kwargs.get('train_data', None)
    save_after = kwargs.get('save_after', 10)
    loss_file_path = kwargs.get('loss_file_path', None)
    model_file_path = kwargs.get('model_file_path', None)
    scaling_factor = kwargs.get('scaling_factor', 0.2)
    fid_file_path = kwargs.get('fid_file_path', None)
    
    if not os.path.exists(loss_file_path):
        os.makedirs(loss_file_path)
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)
    
    if train_data is None:
        raise ValueError("train_data is required")
    
    with open(loss_file_path, 'w') as f:
        f.write(f'Epoch \t Rec_Loss \t KL_Loss \t Combined_Loss\n')
        
    if model_path is None :
        vae = VAE(z_dim=z_dim, layer_count=layer_count)
        vae.weight_init(mean=0.0, std=0.02)
    else:
        vae = torch.load(model_path)

    vae = vae.to(device)
    vae.train()

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    for epoch in range(train_epoch):
        
        # epoch = epoch + 2000
        vae.train()
        rec_loss = 0
        kl_loss = 0
        combined_loss = 0

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        tqdm_batch = tqdm(train_loader, desc=f'VAE training: Epoch {epoch}/{train_epoch}')
        for data, _ in tqdm_batch:
            data = data.to(device)
            vae_optimizer.zero_grad()
            rec, mu, logvar = vae(data)

            loss_re, loss_kl = loss_function(rec, data, mu, logvar, scaling_factor)
            # print("loss_re", loss_re)
            # print("loss_re", loss_kl)
            if(scaling_factor == 0):
                loss_re.backward()
                rec_loss += loss_re.item()
            else:
                (loss_re + loss_kl).backward()
                rec_loss += loss_re.item()
                kl_loss += loss_kl.item()
            vae_optimizer.step()
            
            tqdm_batch.set_postfix(rec_loss=rec_loss, kl_loss=kl_loss)
        print(f'Epoch {epoch}/{train_epoch} Loss: {combined_loss}')
 
        combined_loss = rec_loss + kl_loss
        if(kl_loss != 0 and scaling_factor != 0):
            kl_loss = kl_loss/scaling_factor
        
        if (epoch + 1) % save_after == 0:
            torch.save(vae, f'{model_file_path}/vae_{epoch}.pth')
          
        with open(loss_file_path, 'a') as f:
            f.write(f'{epoch} \t {rec_loss:.4f} \t {kl_loss:.4f} \t {combined_loss:.4f}\n')
        
        with open(fid_file_path,'a') as f:
            fid_score = compute_fid(vae, data_folder_aug_butterfly, z_dim)
            f.write(f"Epoch {epoch}, FID Score: {fid_score}\n")
            f.flush()


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256,1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(z))

class AAE(nn.Module):
    def __init__(self, z_dim=128, layer_count=3, channels=3, d=128, mul=1):
        super(AAE, self).__init__()
        
        self.d = d
        self.z_dim = z_dim
        print(self.z_dim)
        self.layer_count = layer_count

        # Encoder Network
        self.mul = mul
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, self.d * self.mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * self.mul))
            inputs = self.d * self.mul
            self.mul *= 2

        self.d_max = inputs
        self.fc = nn.Linear(inputs * 4 * 4, self.z_dim)

        # Decoder Network
        self.d1 = nn.Linear(self.z_dim, inputs * 4 * 4)
        self.mul = inputs // self.d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d * self.mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(self.d *self. mul))
            inputs = self.d * self.mul
            self.mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

        self.discriminator = Discriminator(z_dim)

    def encode(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        x = x.view(x.shape[0], self.d_max * 4 * 4)
        z = self.fc(x)
        return z

    def decode(self, z):
        x = z.view(z.shape[0], self.z_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 4, 4)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        z = self.encode(x)

        return self.decode(z)
    
    def discriminator_output(self, z):
        return self.discriminator(z)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, res_channels, n_res_layers):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList([ResidualLayer(in_channels, res_channels) for _ in range(n_res_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, h_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1)
        self.res_stack = ResidualStack(h_dim, res_h_dim, n_res_layers)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return self.res_stack(x)

class Decoder(nn.Module):
    def __init__(self, h_dim, n_res_layers, res_h_dim, out_channels):
        super(Decoder, self).__init__()
        self.res_stack = ResidualStack(h_dim, res_h_dim, n_res_layers)
        self.conv1 = nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(h_dim // 2, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.res_stack(x)
        x = F.relu(self.conv1(x))
        return torch.sigmoid(self.conv2(x))

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.beta = beta
        
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.n_embeddings, 1/self.n_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Compute distances between z and embedding vectors
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        
        # Get the closest embedding for each element in z
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = torch.index_select(self.embeddings.weight, dim=0, index=indices.view(-1))
        z_q = z_q.view(z.shape)
        
        # Compute loss
        embedding_loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach() 
        
        return z_q, embedding_loss

class VQVAE(nn.Module):
    def __init__(self, in_channels, h_dim, n_res_layers, res_h_dim, n_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, h_dim, n_res_layers, res_h_dim)
        self.quantizer = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(h_dim, n_res_layers, res_h_dim, in_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, loss


def train_aae(aae_model, discriminator, train_data, num_epochs=100, lr=0.0002, z_dim=128, batch_size = 64,device='cuda', save_after=100, model_file_path = None, loss_file_path = None, fid_file_path = None):
    # Optimizers for both the AAE and the discriminator
    optimizer_aae = optim.Adam(aae_model.parameters(), lr=lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr*0.5)
    
    reconstruction_criterion = nn.MSELoss()  # Could be nn.BCELoss() or nn.L1Loss() as well
    adversarial_criterion = nn.BCELoss()     # For the adversarial loss

    aae_model = aae_model.to(device)
    discriminator = discriminator.to(device)
    real_label, fake_label = 1, 0


    if not model_file_path or not loss_file_path:
        raise ValueError("model_file_path or loss_file_path is required")
    

    with open(loss_file_path, 'a') as f:
        f.write(f'Epoch \t Recon_Loss \t Discriminator_Loss \t Generator_Loss\n')

    for epoch in range(num_epochs):
        epoch+=155
        aae_model.train()
        total_recon_loss = 0
        total_discriminator_loss = 0
        total_generator_loss = 0

        # Loop through batches in the dataset

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        tqdm_batch = tqdm(train_loader, desc=f'AAE training: Epoch {epoch}/{num_epochs}')
       
        for batch,_ in tqdm_batch:
            inputs = batch.to(device)
            batch_size = inputs.size(0)

            # Phase 1: Reconstruction Phase
            optimizer_aae.zero_grad()

            z_encoded = aae_model.encode(inputs)
            recon_images = aae_model.decode(z_encoded)
            recon_loss = reconstruction_criterion(recon_images, inputs)

            # Backpropagation for the AAE (encoder + decoder)
            recon_loss.backward()
            optimizer_aae.step()

            total_recon_loss += recon_loss.item()

            # Phase 2: Regularization Phase (Adversarial Training)

            # Train Discriminator
            optimizer_discriminator.zero_grad()

            # Real latent vectors (sampled from normal distribution)
            z_real = torch.randn(batch_size, z_dim).to(device)
            real_output = discriminator(z_real)
            real_loss = adversarial_criterion(real_output, torch.ones_like(real_output) )

            # Fake latent vectors (encoded by the encoder)
            z_fake = aae_model.encode(inputs).detach()  # Detach so we don't update encoder here
            fake_output = discriminator(z_fake)
            fake_loss = adversarial_criterion(fake_output, torch.zeros_like(fake_output))

            # Total loss for the discriminator
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            optimizer_discriminator.step()

            total_discriminator_loss += discriminator_loss.item()
            for _ in range(2):  # Train the generator (encoder) twice
                optimizer_aae.zero_grad()

                # Forward pass with the generator (encoder)
                z_fake = aae_model.encode(inputs)
                fake_output = discriminator(z_fake)
                generator_loss = adversarial_criterion(fake_output, torch.ones_like(fake_output))

                # Backpropagation for generator loss
                generator_loss.backward()
                optimizer_aae.step()

                total_generator_loss += generator_loss.item()

        # Print losses for the epoch
        print(f"Epoch {epoch}/{num_epochs}, Recon Loss: {total_recon_loss/len(train_loader):.4f}, "
              f"Discriminator Loss: {total_discriminator_loss/len(train_loader):.4f}, "
              f"Generator Loss: {total_generator_loss/len(train_loader):.4f}")
        
        if (epoch + 1) % save_after == 0:
            torch.save(aae_model, f'{model_file_path}/aae_{epoch}.pth')
        
        with open(loss_file_path, 'a') as f:
            f.write(f'{epoch} \t {total_recon_loss/len(train_loader):.4f} \t {total_discriminator_loss/len(train_loader):.4f} \t {total_generator_loss/len(train_loader):.4f}\n')

        with open(fid_file_path,'a') as f:
            fid_score = compute_fid(aae_model, data_folder_aug_butterfly, z_dim)
            f.write(f"Epoch {epoch}, FID Score: {fid_score}\n")
            f.flush()
        
def train_vqvae(model, train_data, num_epochs=100, lr=0.001, batch_size=128, device='cuda', save_after=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_file_path = '/home/siddharth/siddharth/ADRL/Assignment 2/Losses/q7_vqvae/loss.txt'
    model_folder_path = '/home/siddharth/siddharth/ADRL/Assignment 2/Models/q7_vqvae'
    with open(loss_file_path, 'a') as f:
        f.write(f'Epoch \t Total_Loss\n')

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle = True)
  
        tqdm_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch,_ in tqdm_loader:
            inputs = batch.to(device)

            optimizer.zero_grad()

            reconstructions, quantization_loss = model(inputs)

            # Reconstruction loss (MSE)
            reconstruction_loss = F.mse_loss(reconstructions, inputs)

            # Total loss
            loss = reconstruction_loss + quantization_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
        with open(loss_file_path, 'a') as f:
            f.write(f'{epoch} \t {avg_loss}')
        if (epoch+1)%save_after == 0 :
            torch.save(model, f'{model_folder_path}/vq_vae_{epoch}.pth')
        

        


train_dataset_butterfly = ButterflyDataset(data_folder_aug_butterfly)
train_dataset_animals = AnimalsDataset(data_folder_aug_animals)

'''
#Usag
in_channels = 3
h_dim = 256
n_res_layers = 2
res_h_dim = 64
n_embeddings = 512
embedding_dim =256  # Should be same as h_dim
beta = 0.25
batch_size = 128
num_epochs = 500
lr=0.001

model = VQVAE(in_channels=in_channels, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim, n_embeddings=n_embeddings, embedding_dim=embedding_dim, beta=beta)
train_data = ButterflyDataset(data_folder_aug_butterfly)
train_vqvae(model, train_data, num_epochs, lr, batch_size)'''



args = {'z_dim': 512,
        'batch_size': 128,
        'train_epoch': 2000,
        'pretrained_model_path': None,
        'lr': 0.0005,
        'layer_count': 5,
        'device': 'cuda',
        'train_data': train_dataset_butterfly,
        'save_after': 50,
        'scaling_factor':1,
        'loss_file_path':"/home/siddharth/siddharth/ADRL/Assignment 2/Losses/q1_vae/t5.txt",
        'model_file_path':"/home/siddharth/siddharth/ADRL/Assignment 2/Models/q1_vae/q1_vae_bf_t2",
        'fid_file_path':"/home/siddharth/siddharth/ADRL/Assignment 2/FIDs/q1_vae/t5.txt"
        }

train(**args)


##################----AAE-----#########################################################
# aae_model = AAE(z_dim=128, layer_count=5, channels=3)
'''aae_model = torch.load("/home/siddharth/siddharth/ADRL/Assignment 2/Models/aae/aae_155.pth")
aae_model.weight_init(mean = 0.0, std = 0.02)

num_epochs = 400
lr = 0.0002
z_dim = 128
batch_size = 64
device = "cuda"
save_after = 40
model_file_path= "/home/siddharth/siddharth/ADRL/Assignment 2/Models/aae"
loss_file_path = "/home/siddharth/siddharth/ADRL/Assignment 2/Losses/aae_losses/aae_1.txt"
fid_file_path = "/home/siddharth/siddharth/ADRL/Assignment 2/FIDs/aae/fid_1.txt"
train_dataset_butterfly = ButterflyDataset(data_folder_aug_butterfly)
disciminator = Discriminator(aae_model.z_dim)
train_aae(aae_model, discriminator=disciminator, train_data = train_dataset_butterfly, num_epochs=num_epochs, lr = lr, z_dim = z_dim, batch_size=batch_size, device = device, save_after=save_after, model_file_path=model_file_path, loss_file_path=loss_file_path, fid_file_path=fid_file_path)
'''
