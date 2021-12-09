from visualize_method import visualize_grid
import argparse
import os
import torch 
import torchvision.transforms as transforms 
from dataset import CovidDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

city = 'csv/nyc'
image_folder = 'proj/vondrick/datasets/EarthCam/nyc_photos'
parser = argparse.ArgumentParser(description="training pipeline") 

parser.add_argument('--attributes_file', type=str, default='./' + city + '_photos.csv')

parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

start_epoch = 1
N_epochs = 50
batch_size = 16
num_workers = 8
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu") 
attributes = AttributesDataset(args.attributes_file) 


# specify image transforms for augmentation during training
train_transform = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.5),
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
			shear=None, resample=False, fillcolor=(255, 255, 255)),
transforms.ToTensor(),
transforms.Normalize(mean, std)
])

# during validation we use only tensor and normalization transforms
val_transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean, std)
])

train_dataset = CovidDataset('./train.csv', attributes, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = CovidDataset('./val.csv', attributes, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = MultiOutputModel(n_month_classes=attributes.num_months,
		     n_season_classes=attributes.num_seasons,
		     n_hour_classes=attributes.num_hours)\
		    .to(device)

optimizer = torch.optim.Adam(model.parameters())

visualize_grid(model, val_dataloader, attributes,device, show_cn_matrices=True, show_images=False, checkpoint='checkpoints/2021-03-22_13-57/checkpoint-000050.pth', show_gt=False)
