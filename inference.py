import os
import requests
import pandas as pd
import cv2
import tqdm

URL = 'http://10.10.66.129:5010/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

def run_service(validate_path: str):
    metadata = pd.DataFrame(columns=['x', 'y'])
    image_files = [i for i in os.listdir(validate_path) if i.endswith(('.jpg', '.jpeg', '.png'))]
    # final = [i for i in image_files if i != "037.jpg"]
    # import pdb;pdb.set_trace()
    for image_name in tqdm.tqdm(image_files):
        metadata = pd.DataFrame(columns=['x', 'y'])
        image_array = cv2.imread(os.path.join(validate_path, image_name))
        # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', image_array)
        data = img_encoded.tostring()
        response = requests.post(URL, data=data, headers=headers)
        recognitions = response.json()['image']
        
        for bbox in recognitions['bbox']:
            
            topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
            botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])
            center_coords = int((botRightCorner[0] + topLeftCorner[0]) / 2), int((botRightCorner[1] + topLeftCorner[1]) / 2)

            metadata = metadata.append({'x':center_coords[0], 'y':center_coords[1]}, ignore_index=True)
        metadata.to_csv(f"tests_results/{image_name.split('.jpg')[0]}.csv", index=False)
    return metadata


if __name__=="__main__":
    #--------------------Параметры-для-изменения--------------------#
    dataset_path = '/media/storage_1/habarovsk_hack/hui/'
    #---------------------------------------------------------------#
    
    res = run_service(dataset_path)
    # res.to_csv('test.csv', index=False)