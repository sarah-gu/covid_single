import torch
import torch.nn.functional as F

from PIL import Image

from test import checkpoint_load
from dataset import CovidDataset, AttributesDataset, mean,std
from model import MultiOutputModel

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

import captum
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

city = 'nyc'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributes = AttributesDataset('./csv/' + city + '_photos.csv')

model = MultiOutputModel(n_month_classes=attributes.num_months, n_season_classes=attributes.num_seasons, n_hour_classes=attributes.num_hours).to(device)

checkpoint_load(model, "checkpoints/2021-03-01_23-50/checkpoint-000050.pth")
model = model.eval()
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean,std) ])

val_dataset = CovidDataset('./val.csv', attributes, transform)
val_dataloader = DataLoader(val_dataset, batch_size = 16, shuffle=False, num_workers=8)

with torch.no_grad(): 
	
	for batch in dataloader: 
		img = batch['img']
		img_path = batch['img_path']
		
		output = model(img.to(device))
		
		_, predicted_months = output['month'].cpu().max(1)
		_, predicted_seasons = output['season'].cpu().max(1)
		_, predicted_hours = output['hour'].cpu().max(1)

		for i in range(img.shape[0]):
			predicted_month = predicted_months[i].item()
			predicted_season = predicted_seasons[i].item()
			predicted_hour = predicted_hours[i].item()
			
			im = Image.open(img_path)
			transformed_img = transform(im)
			input = img.unsqueeze(0)
			
			integrated_gradients = IntegratedGradients(model)
			attributions_ig = integrated_gradients.attribute(input, target=predicted_month, n_steps=200)
			default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

			_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)

			noise_tunnel = NoiseTunnel(integrated_gradients)

			attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=predicted_month)

			_ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)

			torch.manual_seed(0)
			np.random.seed(0)

			gradient_shap = GradientShap(model)

# Defining baseline distribution of images
			rand_img_dist = torch.cat([input * 0, input * 1])

			attributions_gs = gradient_shap.attribute(input,
                                          n_samples=50,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=pred_label_idx)
			_ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)


