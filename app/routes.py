from .model import load_mmdet_model, load_metric_model, get_metric_prediction
from flask import request, render_template
import numpy as np
import cv2
from app import app
import torch
import random
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

metric_model_path = os.getenv('METRIC_MODEL')
embs = os.getenv('EMBEDDINGS')

detect_model_path = os.getenv('DETECTION_MODEL_RCNN')


def get_image_from_tg_bot(request_from_bot):
    """Функция для обрабтки фотографии с тг запроса
    Args:
        request_from_bot: bytearray - байтовое представление изображения
    Return:
        img: np.array - numpy массив-представление изображения
    """

    nparr = np.fromstring(request_from_bot.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


mmdet_model = load_mmdet_model(detect_model_path, threshold=0.83) # выгрузка детектора
metric_model, feature_extractor, device, base = load_metric_model(metric_model_path, embs) # выгрузка metric learning модели


@app.route('/')
@app.route('/index')
def index():
    """Рендеринг html страницы

    Показательная страница, показывающая возможность простого расширения и внедрения сервиса
    """

    return render_template('index.html')

@app.route('/api/test', methods=['POST'])
def test():
    print("backend start")
    img = get_image_from_tg_bot(request)
    print(img.shape)
    enhancer = ImageEnhance.Contrast(Image.fromarray(img))
    img = np.array(enhancer.enhance(1.5))
    print(img.shape)


    detection_result = get_sliced_prediction(img, mmdet_model, slice_height = 1024, slice_width = 1024).to_coco_predictions()


    # if len(detection_result) > 200:
    #     detection_result = get_sliced_prediction(img, detection_model, slice_height = 512, slice_width = 512).to_coco_predictions()

    print('detecting is done')

    all_bboxes = []
    for i, recognition in enumerate(detection_result):

        if len(recognition['bbox']) == 0:
            continue

        all_bboxes.append({'bbox_id':i, 
                            'bbox':{'x':int(recognition['bbox'][0]), 'y':int(recognition['bbox'][1]),\
                                    'width':int(recognition['bbox'][2]), 'height':int(recognition['bbox'][3])},\
                            'confidence':int(recognition['score']*100),
                            'class_name': 'walrus'})
    
    for bbox in tqdm(all_bboxes):

        topLeftCorner = (bbox['bbox']['x'], bbox['bbox']['y'])
        botRightCorner = (bbox['bbox']['x']+bbox['bbox']['width'], bbox['bbox']['y']+bbox['bbox']['height'])
        
        # import pdb; pdb.set_trace() 
        cutted_img = img[topLeftCorner[1]:botRightCorner[1], topLeftCorner[0]:botRightCorner[0]]

        metric_result = get_metric_prediction(metric_model, feature_extractor, device, base, cutted_img)

        classes_dict = {0: 'leopard', 1: 'princess', 2: 'tigers', 3: 'other animal'}

        bbox.update({'class_name': classes_dict[metric_result]})
        print(metric_result)

    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox':all_bboxes}
                }

    return response