import os
import cv2
from tqdm import tqdm
from mmcv import Config
import mmcv
from mmdet.apis import init_detector, inference_detector
import numpy as np
import torch
import random

from utils import draw_contours

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def load_detection_model():
    """Функция для загрузки обученной модели детекции"""

    cfg = Config.fromfile('source/hackaton_0.652map_config.py')

    checkpoint_path = 'source/epoch_20.pth'
    detector_model = init_detector(cfg, checkpoint=checkpoint_path, device='cpu')
    return detector_model
    
def get_detection_prediction(model, img):
    """"Функция для инференса детектора
    
    Args:
        model - выгруженная модель детектора
        img: np.array - полученное изображение с запроса 
    Return:
        result - предикт модели (состоит из bbox с координатами и вероятностью)
    """

    result = inference_detector(model, img)
    all_bboxes = []
    for i, recognition in enumerate(result):

        if recognition.shape[0] == 0: # проверка результата детектора, если по [0] ничего, мы его пропускаем
            continue

        for bbox in recognition:
            if bbox[-1] > 0.45: # подобранный threshold для точности детекции
                all_bboxes.append({'bbox_id':i, 'bbox':{'x1':int(bbox[0]), 'y1':int(bbox[1]),\
                                                        'x2':int(bbox[2]), 'y2':int(bbox[3])},\
                                   'threshold':int(bbox[-1]*100),
                                   'class_name': 'walrus'})
    return all_bboxes

if __name__ == "__main__":

    predictor = load_detection_model()
    test_dir = 'test-public-images'
    predict_dir = 'submission'
    for file_path in tqdm(os.listdir(test_dir)):
        
        image = mmcv.imread(f'{test_dir}/{file_path}')
        result = get_detection_prediction(predictor, image)
        count_class = draw_contours(image, result)
        cv2.imwrite(f'{predict_dir}/{file_path}', image)