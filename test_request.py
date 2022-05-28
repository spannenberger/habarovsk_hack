import cv2
import os
import requests
import pandas as pd
import tqdm

# Для запуска на вашей машине нужно поменять URL на ваш локальный ip 
URL = 'http://localhost:5000/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}
def test_req():
    image_array = cv2.imread('test-public-images/photo_2022-05-24.jpeg')
    _, img_encoded = cv2.imencode('.jpg', image_array)
    data = img_encoded.tostring()
    response = requests.post(URL, data=data, headers=headers)
    metadata = response.json()['image']
    return metadata


if __name__=="__main__":
    #--------------------Параметры-для-изменения--------------------#
    dataset_path = './test_data/'
    #---------------------------------------------------------------#
    print(test_req())
    # get_recognize_leotigers(dataset_path) # Раскомментировать и запустить повторно