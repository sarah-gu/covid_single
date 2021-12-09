import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AttributesDataset():
    def __init__(self, annotation_path):
        month_labels = []
#        season_labels = []
 #       hour_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                month_labels.append(row['month'])
  #              season_labels.append(row['season'])
  #              hour_labels.append(row['hour'])

        self.month_labels = np.unique(month_labels)
   #     self.season_labels = np.unique(season_labels)
   #     self.hour_labels = np.unique(hour_labels)

        self.num_months = len(self.month_labels)
   #     self.num_seasons = len(self.season_labels)
   #     self.num_hours = len(self.hour_labels)

        self.month_id_to_name = dict(zip(range(len(self.month_labels)), self.month_labels))
        self.month_name_to_id = dict(zip(self.month_labels, range(len(self.month_labels))))

  #      self.season_id_to_name = dict(zip(range(len(self.season_labels)), self.season_labels))
   #     self.season_name_to_id = dict(zip(self.season_labels, range(len(self.season_labels))))

   #     self.hour_id_to_name = dict(zip(range(len(self.hour_labels)), self.hour_labels))
   #     self.hour_name_to_id = dict(zip(self.hour_labels, range(len(self.hour_labels))))


class CovidDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.month_labels = []
    #    self.season_labels = []
    #    self.hour_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['id'])
                self.month_labels.append(self.attr.month_name_to_id[row['month']])
     #           self.season_labels.append(self.attr.season_name_to_id[row['season']])
     #           self.hour_labels.append(self.attr.hour_name_to_id[row['hour']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'img_path': img_path,
            'labels': {
                'month_labels': self.month_labels[idx],
     #           'season_labels': self.season_labels[idx],
     #           'hour_labels': self.hour_labels[idx]
            }
        }
        return dict_data
