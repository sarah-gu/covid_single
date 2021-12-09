import csv
import os 
city ='nyc'
metadata_folder = '/proj/vondrick/datasets/EarthCam/' +city + '_metadata'
image_folder = '/proj/vondrick/datasets/EarthCam/' + city + '_photos'

months = {"Jan": 1, "Feb":2, "Mar":3, "Apr": 4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep": 9, "Oct":10, "Nov":11, "Dec": 12}

images = [img for img in os.listdir(image_folder)]
metadata = [txt for txt in os.listdir(metadata_folder)]

with open( city + '_photos.csv', mode='w') as outfile: 
    out_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    out_writer.writerow(['id', 'month', 'season', 'hour'])
    for txt in metadata: 
        f = open(os.path.join(metadata_folder,txt), "r")
        line = f.read().split(" ")
        img_id = txt[:-4] 
        img_url = img_id + '.jpg'

        if line[0] != '':
            hourmin = line[-1][:-2].split(":")
            hour = int(hourmin[0])
            if 'am' in line[-1] and hour == 12:
                hour = 0
            if 'pm' in line[-1] and hour != 12:
                hour = hour + 12
        
            season = ''
            month = months[line[0]]
            if month > 2 and month < 6: 
                season = 'spring'
            elif month > 5 and month < 9: 
                season = 'summer'
            elif month > 8 and month < 12: 
                season = 'fall'
            else: 
                season = 'winter' 

            out_writer.writerow([img_id, line[0], season, str(hour)])






