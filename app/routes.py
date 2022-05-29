from .model import load_mmdet_model, load_metric_model, get_metric_prediction
from .utils import image_augmentations, get_image_from_tg_bot
from sahi.predict import get_sliced_prediction
from flask import request, render_template
from tqdm import tqdm
from app import app
import numpy as np
import random
import torch
import os


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

metric_model_path = os.getenv('METRIC_MODEL', '')
embs = os.getenv('EMBEDDINGS', '')

detect_model_rcnn_path = os.getenv('DETECTION_MODEL_RCNN', '')
detect_model_rcnn_hr_path = os.getenv('DETECTION_MODEL_RCNN_HR', '')


# mmdet_rcnn_model = load_mmdet_model(detect_model_rcnn_path, threshold=0.92) # выгрузка детектора
mmdet_rcnn_hr_model = load_mmdet_model(detect_model_rcnn_hr_path, threshold=0.85) # выгрузка детектора

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

    img = image_augmentations(img)

    # detection inference using SAHI
    # detection_result_rcnn = get_sliced_prediction(img, mmdet_rcnn_model, slice_height = 1024, slice_width = 1024).object_prediction_list
    detection_result_rcnn_hr = get_sliced_prediction(img, mmdet_rcnn_hr_model, slice_height = 1024, slice_width = 1024).object_prediction_list

    print('detecting is done')

    all_bboxes = []

    for i, response in tqdm(enumerate(detection_result_rcnn_hr)):

        confidence = response.score.value

        if len(response.bbox.to_voc_bbox()) == 0:
            continue

        response = response.bbox.to_voc_bbox()

        all_bboxes.append({'bbox_id': i,
                            'bbox': {
                                'x1': response[0], 
                                'y1': response[1],
                                'x2': response[2], 
                                'y2': response[3]},
                            'confidence': int(confidence * 100),
                            'class_name': 'walrus'})
    
    for bbox in tqdm(all_bboxes):

        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])
        
        cutted_img = img[topLeftCorner[1]:botRightCorner[1], topLeftCorner[0]:botRightCorner[0]]

        metric_result = get_metric_prediction(metric_model, feature_extractor, device, base, cutted_img)

        classes_dict = {0: 'leopard', 1: 'princess', 2: 'tigers', 3: 'other animal'}

        bbox.update({'class_name': classes_dict[metric_result]})

    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox':all_bboxes}
                }

    return response