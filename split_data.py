import csv
from tqdm import tqdm 
import os 
from PIL import Image
import numpy as np
import cv2 

city = 'nyc'
csv_file = 'csv/' + city + '_photos.csv'
img_path = '/proj/vondrick/datasets/EarthCam/' + city + '_photos/'


all_data = []
def save_csv(arr, filename): 
    
    with open(filename, mode='w') as outfile: 
        out_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        out_writer.writerow(['id', 'month', 'season', 'hour'])
        for entry in arr: 
#            print(entry)
            out_writer.writerow(entry)




with open(csv_file) as toparse: 
    reader = csv.DictReader(toparse)

    for row in tqdm(reader, total=reader.line_num): 
        img_id = row['id']
        month = row['month']
        season = row['season']
        hour = row['hour']

        img_name = os.path.join(img_path, img_id + '.jpg')
        if os.path.exists(img_name): 
            ogimg = cv2.imread(img_name, cv2.IMREAD_COLOR)
 #           print(img_name)
            #shape = (ogimg.shape)
#            img = Image.open(img_name)
            if ogimg.shape == (540, 960, 3): 
            	all_data.append([img_name, month, season, hour])
        else: 
            print("Something went wrong")


    
np.random.seed(18)
img_length = len(all_data)
all_data = np.asarray(all_data)

inds = np.random.choice(img_length, img_length, replace=False)
print(all_data[inds][:int(img_length * .85)])
save_csv(all_data[inds][:int(img_length * .85)], 'train.csv')
save_csv(all_data[inds][int(img_length * .85): img_length], 'val.csv')

