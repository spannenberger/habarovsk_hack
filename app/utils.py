from PIL import Image, ImageEnhance
import numpy as np
import cv2


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

def image_augmentations(img: np.array):
    """Добавление контраста изображению 
    Args:
        img: np.array - исходное изображение
    Return:
        img: np.array - аугментированное изображение
    """

    enhancer = ImageEnhance.Contrast(Image.fromarray(img))
    img = np.array(enhancer.enhance(1.5))

    return img