from typing import Dict
import requests
import cv2

# Для запуска на вашей машине нужно поменять URL на ваш локальный ip 
URL = 'http://10.10.66.54:5010/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

def test_req(img_path: str) -> Dict:
    image_array = cv2.imread(img_path)
    _, img_encoded = cv2.imencode('.jpg', image_array)
    data = img_encoded.tostring()

    response = requests.post(URL, data=data, headers=headers)
    metadata = response.json()['image']
    
    return metadata

def main():
    img_path = 'test-public-images/photo_2022-05-29 05.40.18.jpeg'
    res = test_req(img_path)
    print(len(res['bbox']))

if __name__=="__main__":
    main()