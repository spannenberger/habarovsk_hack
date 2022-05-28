from tqdm import tqdm
import numpy as np
import random
import torch
import mmcv
import cv2
import os

from utils import load_detection_model, get_detection_prediction

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def draw_contours(image_array, metadata):
    """ Функция отрисовки контура и подсчета кол-во особей
    Обрабатываем результат работы моделей, извлекая полученный класс животного
    Args:
        image_array: arr - массив-представление изображения
        metadata: json - словарь, содержащий ответ работы моделей
    Return:
        counter_dict: dict - словарь с кол-вом определенных животных
    """

    counter = len(metadata)
    for bbox in metadata:
        class_name = bbox['class_name']

        threshold = bbox['threshold']
        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

        cv2.rectangle(image_array,\
                         topLeftCorner,\
                         botRightCorner,\
                         (255, 0, 0), 1)

        cv2.putText(image_array, f'{class_name} - {threshold}',
                        topLeftCorner,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        2)

    return counter

if __name__ == "__main__":

    predictor = load_detection_model('source/hackaton_0.652map_config.py', 'source/epoch_20.pth')
    test_dir = 'test-public-images'
    predict_dir = 'submission'
    for file_path in tqdm(os.listdir(test_dir)):
        
        image = mmcv.imread(f'{test_dir}/{file_path}')
        result = get_detection_prediction(predictor, image)
        count_class = draw_contours(image, result)
        print(count_class)
        cv2.imwrite(f'{predict_dir}/{file_path}', image)