import io
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmcv import Config

def load_detection_model(config: str, model_path: str):
    """Функция для загрузки обученной модели детекции"""

    cfg = Config.fromfile(config)

    checkpoint_path = model_path
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
            # print(bbox[-1])
            if bbox[-1] > 0.45: # подобранный threshold для точности детекции
                all_bboxes.append({'bbox_id':i, 'bbox':{'x1':int(bbox[0]), 'y1':int(bbox[1]),\
                                                        'x2':int(bbox[2]), 'y2':int(bbox[3])},\
                                   'threshold':int(bbox[-1]*100),
                                   'class_name': 'walrus'})
    return all_bboxes


def transform_pil_image_to_bytes(image):
    """Перевод из array в изображение

    @TODO DOCS
    Args:
        image: np.array - представление фотографии
    Return:
        buffer: bytes - сохраненная фотография
    """

    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)

    return buffer