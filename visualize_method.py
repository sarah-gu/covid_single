from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import PIL
from test import checkpoint_load, validate, calculate_metrics
import argparse
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import CovidDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay

city = 'csv/nyc'
image_folder = '/proj/vondrick/datasets/EarthCam/nyc_photos'

font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Black.ttf", size=25, encoding="unic")
def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_month_all = []
    gt_season_all = []
    gt_hour_all = []
    predicted_month_all = []
    predicted_season_all = []
    predicted_hour_all = []

    accuracy_month = 0
    accuracy_season = 0
    accuracy_hour = 0
    frame = cv2.imread(os.path.join(image_folder, '1589747168328_67.jpg'))
    height, width, layers = frame.shape
    video = cv2.VideoWriter('output.avi', 0, .5, (width,height))
    count = 0
    with torch.no_grad():
       num = 0      
       for batch in dataloader:
            img = batch['img']
            img_path = batch['img_path']
            gt_months = batch['labels']['month_labels']
            gt_seasons = batch['labels']['season_labels']
            gt_hours = batch['labels']['hour_labels']
            output = model(img.to(device))
            
            batch_accuracy_month, batch_accuracy_season, batch_accuracy_hour = \
                calculate_metrics(output, batch['labels'])
            accuracy_month += batch_accuracy_month
            accuracy_season += batch_accuracy_season
            accuracy_hour += batch_accuracy_hour

            # get the most confident prediction for each image
            _, predicted_months = output['month'].cpu().max(1)
            _, predicted_seasons = output['season'].cpu().max(1)
            _, predicted_hours = output['hour'].cpu().max(1)
            
            for i in range(img.shape[0]):
                #print(img[i])
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)
                #pil = transforms.ToPILImage()().convert("RGB")
                #forcv2 = np.array(pil)
                #print(image)
                #forcv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)         
                #forcv2 = cv2.normalize(forcv2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#      forcv2 = forcv2[:,:,::-1].copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                im = cv2.imread(img_path[i])
     
                predicted_month = attributes.month_id_to_name[predicted_months[i].item()]
                predicted_season = attributes.season_id_to_name[predicted_seasons[i].item()]
                predicted_hour = attributes.hour_id_to_name[predicted_hours[i].item()]
#                print(predicted_month)
      #          predicted_month = predicted_months[i].item()
       #         predicted_season = predicted_seasons[i].item()
        #        predicted_hour = predicted_hours[i].item()

                gt_month = attributes.month_id_to_name[gt_months[i].item()]
                gt_season = attributes.season_id_to_name[gt_seasons[i].item()]
                gt_hour = attributes.hour_id_to_name[gt_hours[i].item()]
  #              gt_month = gt_months[i].item()
   #             gt_season = gt_seasons[i].item()
    #            gt_hour = gt_hours[i].item()
                
                if count %100 == 0: 
                     #im = Image.fromarray((image * 255).astype(np.uint8))
                     #im.save('images/test' + str(num) + '.png')
                     #im = Image.open("images/test" + str(num) + ".png")
                     #draw = ImageDraw.Draw(im)
                     #toWrite = gt_month + ", " + gt_season + ", " + gt_hour + ", predicted: " + predicted_month + ", " + predicted_season + ", " + predicted_hour                     
                     #draw.text((0,0), toWrite, (255,255,0), font=font)
                     #im = Image.fromarray(image)
                     cv2.putText(im, gt_month + ", " + gt_season + ", " + gt_hour + ", predicted: " + predicted_month + ", " + predicted_season + ", " + predicted_hour, (10,450), font, 1, (0,255,0), 2, cv2.LINE_AA)
                     #im.save("images/test" + str(num) + ".png")
                     #num += 1
                     video.write(im)
                gt_month_all.append(gt_month)
                gt_season_all.append(gt_season)
                gt_hour_all.append(gt_hour)
    #            print(gt_month_all)
                predicted_month_all.append(predicted_month)
                predicted_season_all.append(predicted_season)
                predicted_hour_all.append(predicted_hour)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_season, predicted_hour, predicted_month))
                gt_labels.append("{}\n{}\n{}".format(gt_season, gt_hour, gt_month))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nmonth: {:.4f}, season: {:.4f}, hour: {:.4f}".format(
            accuracy_month / n_samples,
            accuracy_season / n_samples,
            accuracy_hour / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # month
        #print(attributes.month_labels)
        #print(gt_month_all)
        month_labels = ['Jan', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        cn_matrix = confusion_matrix(
            y_true=gt_month_all,
            y_pred=predicted_month_all,
            labels = month_labels, normalize="true"      
            )
        ConfusionMatrixDisplay(cn_matrix, month_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Months")
        plt.tight_layout()
        plt.show()
        plt.savefig('cn_months.png') 

        # season
        cn_matrix = confusion_matrix(
            y_true=gt_season_all,
            y_pred=predicted_season_all,
            labels=attributes.season_labels,normalize="true"
            )
        ConfusionMatrixDisplay(cn_matrix, attributes.season_labels).plot(
            xticks_rotation='horizontal')
        plt.title("Seasons")
        plt.tight_layout()
        plt.show()
        plt.savefig('cn_seasons.png')
        # Uncomment code below to see the hour confusion matrix (it may be too big to display)
        hour_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14','15', '16', '17','18', '19', '20', '21', '22', '23', '24']
        cn_matrix = confusion_matrix(
            y_true=gt_hour_all,
            y_pred=predicted_hour_all,
            labels=hour_labels,normalize="true"
            )
        plt.rcParams.update({'font.size': 1.8})
        plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(cn_matrix, hour_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.rcParams.update({'figure.dpi': 100})
        plt.rcParams.update({'font.size': 5})
        plt.title("Hours")
        plt.show()
        plt.savefig('cn_hours.png') 

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()



