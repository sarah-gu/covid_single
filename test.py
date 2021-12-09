from PIL import Image
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
PHOTOS_PER_DAY = 1
metadata_folder = 'nyc_metadata'

def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_month = 0
        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_month = \
                calculate_metrics(output, target_labels)

            accuracy_month += batch_accuracy_month

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_month /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, month: {:.4f}\n".format(
        avg_loss, accuracy_month))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_month', accuracy_month, iteration)

    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_month_all = []
    predicted_month_all = []

    accuracy_month = 0
    #frame = cv2.imread(os.path.join(image_folder, '1589747168328_67.jpg'))
    #height, width, layers = frame.shape
    #video = cv2.VideoWriter('output.avi', 0, .5, (width,height))
    count = 0
    with torch.no_grad():
      
       for batch in dataloader:
            img = batch['img']
            gt_months = batch['labels']['month_labels']
            output = model(img.to(device))
            
            batch_accuracy_month= \
                calculate_metrics(output, batch['labels'])
            accuracy_month += batch_accuracy_month

            # get the most confident prediction for each image
            _, predicted_months = output.cpu().max(1)
            
            for i in range(img.shape[0]):
    
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)
    #            pil = transforms.ToPILImage()(img[i]).convert("RGB")
     #           forcv2 = np.array(pil)
      #          forcv2 = forcv2[:,:,::-1].copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
               # im = cv2.imread(os.path.join(image_folder, image))
     
                predicted_month = attributes.month_id_to_name[predicted_months[i].item()]
#                print(predicted_month)
      #          predicted_month = predicted_months[i].item()
       #         predicted_season = predicted_seasons[i].item()
        #        predicted_hour = predicted_hours[i].item()

                gt_month = attributes.month_id_to_name[gt_months[i].item()]
       #         if count %100 == 0: 
        #             cv2.putText(forcv2, gt_month + ", " + gt_season + ", " + gt_hour + ", predicted: " + predicted_month + ", " + predicted_season + ", " + predicted_hour, (10,450), font, 3, (0,255,0), 2, cv2.LINE_AA)
         #            video.write(forcv2)
                gt_month_all.append(gt_month)
    #            print(gt_month_all)
                predicted_month_all.append(predicted_month)

                imgs.append(image)
                labels.append("{}\n".format(predicted_month))
                gt_labels.append("{}\n".format(gt_month))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nmonth: {:.4f}".format(
            accuracy_month / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # month
        print(attributes.month_labels)
        print(gt_month_all)
        month_labels = ['Jan', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        cn_matrix = confusion_matrix(
            y_true=gt_month_all,
            y_pred=predicted_month_all,
            labels = month_labels       
            )
        ConfusionMatrixDisplay(cn_matrix, month_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Months")
        plt.tight_layout()
        plt.show()
        plt.savefig('cn_months.png') 
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


def calculate_metrics(output, target):
    _, predicted_month = output.cpu().max(1)
    gt_month = target['month_labels'].cpu()

  #  _, predicted_season = output['season'].cpu().max(1)
   # gt_season = target['season_labels'].cpu()

    #_, predicted_hour = output['hour'].cpu().max(1)
   # gt_hour = target['hour_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_month = balanced_accuracy_score(y_true=gt_month.numpy(), y_pred=predicted_month.numpy())
    #    accuracy_season = balanced_accuracy_score(y_true=gt_season.numpy(), y_pred=predicted_season.numpy())
     #   accuracy_hour = balanced_accuracy_score(y_true=gt_hour.numpy(), y_pred=predicted_hour.numpy())

    return accuracy_month


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./' + city + '_photos.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = CovidDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(n_month_classes=attributes.num_months)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)
