from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import torch
import random
from sahi.model import MmdetDetectionModel

# from source.infrastructure.detection.yolov5 import YoloV5Detector
# from source.infrastructure.detection.config import AppConfig, ObjectDetectorConfig

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_mmdet_model(detect_model_path, threshold: float = 0.4):
    """Функция для загрузки обученной модели детекции"""
    # import pdb; pdb.set_trace()
    detector_model = MmdetDetectionModel(
    model_path=f"{detect_model_path}/latest.pth",
    config_path=f'{detect_model_path}/custom_config.py',
    confidence_threshold=threshold,
    device=device)

    return detector_model


# def load_yolo_model():
#     """Функция для загрузки обученной модели детекции"""
#     app_config = AppConfig()
#     detector_config = app_config.object_detector
#     object_detector = YoloV5Detector(detector_config)
#     return object_detector


def load_metric_model(model_path, csv_path):
    """Загрузки обученной модели metric learning 

    Модель metric learning используется для определения принцессы

    Return:
        metric_model - выгруженная metric learning модель
        feature_extractor - выгруженный экстрактор фичей с изображения
        device - девайс на который мы ставим модель
        base: np.array - массив с классами и их усредненными эмбеддингами
    """

    extractor = 'google/vit-base-patch16-384' # фичи экстрактор
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # пока пусть так будет, нужно сделать не так явно

    df = pd.read_csv(csv_path) # считываем csv со средними эмбеддингами для каждого класса
    base = df.values.T

    feature_extractor = ViTFeatureExtractor.from_pretrained(extractor)
    metric_model = ViTModel.from_pretrained(model_path)
    metric_model.to(device)

    return metric_model, feature_extractor, device, base


def get_metric_prediction(
    model, 
    feature_extractor, 
    device, 
    base, 
    img):
    """ Функция инференса metric learning

    Прогоняем фотографию через модель metric learning для получения его эмбеддинга
    Смотрим косинусное расстояние до эталонного эмбеддинга запредикченного класса
    (Далле по найденному threshold мы будем отсекать фотографии где нет принцессы)

    Args:
        model - выгруженная metric learning модель
        feature_extractor - экстрактор фичей с изображения
        device - девайс на который мы ставим модель
        base: np.array - массив с классами и их усредненными эмбеддингами
        img: np.array - полученное изображение с запроса

    Return:
        correct_class: int - близжайший класс, к которому мы отнесли фотографию
    """
    
    img = feature_extractor(img, return_tensors="pt")
    img.to(device)

    # инференс модели и получение предикта
    model.eval()
    with torch.no_grad():
        prediction = model(**img).pooler_output
        prediction = prediction[0].cpu().detach().numpy()

    dist = []
    for emb in base:
        dist.append(cosine(emb, prediction)) # считаем косинусное расстояние

    class_idx = np.argmin(np.array(dist)) # берем индекс наименьшего расстояния - близжайший класс

    # print(dist[class_idx])

    if dist[class_idx] < 0.42:
        correct_class = class_idx

    else:
        correct_class = 3

    return correct_class


def get_detection_prediction(model, img):
    """"Функция для инференса детектора
    
    Args:
        model - выгруженная модель детектора
        img: np.array - полученное изображение с запроса 
    Return:
        result - предикт модели (состоит из bbox с координатами и вероятностью)
    """

    result = inference_detector(model, img)

    return result
