import os
import numpy as np
import torch
from model import MultiOutputModel
from pytorch_grad_cam import GradCAM
import cv2
import argparse
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', action = 'store_true', default = False)
parser.add_argument('--image-path', type=str) 
args = parser.parse_args()
args.use_cuda = args.use_cuda and torch.cuda.is_available()

image_folder = '/proj/vondrick/datasets/EarthCam/prev_photos'
metadata_folder = 'prev_metadata'

images = [img for img in os.listdir(image_folder)]


model = MultiOutputModel(11) 
model.load_state_dict(torch.load('month_50_ckpt.pt'))
print("model loaded") 


frame = cv2.imread(args.image_path)
height, width, layers = frame.shape
video = cv2.VideoWriter('output.avi', 0, 0.5, (width, height))


for img in images:
	target_layers = [model.base_model[-1]]
	rgb_img = cv2.imread(os.path.join(image_folder, img),1)[:, :, ::-1]
	rgb_img = np.float32(rgb_img) / 255
	input_tensor = preprocess_image(rgb_img, mean = [0.485, 0.456, 0.406], std = [0.229, .224, .225])

	cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)

	target_category = None

	grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

	# In this example grayscale_cam has only one image in the batch:
	grayscale_cam = grayscale_cam[0, :]
	visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

	cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
	video.write(cam_image)
	#cv2.imwrite(f'nyc_gradcam.jpg', cam_image)

