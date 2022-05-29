from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import io

def process_image(image_bytes):
    """Перевод изображения из байтов в необходимый для вывода формат
    
    Args: 
        image_bytes: bytearray - битовое представление изображения
    Return:
        image: cv2.cvtColor - само изображение
    """

    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def transform_pil_image_to_bytes(image):
    """Перевод из array в изображение
    
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


def get_image_bytes(bot, update):
    """ Выгрузка полученной фотографии

    Args:
        bot: bot - тело нашего бота
        update: command - параметр для обновления данных в боте
    Return:
        image: bytearray - битовое представление полученного изображения
        file_id: int - уникальный id полученного изображения 
    """

    if not update.message.photo:
        file_id = update.message.document['file_id']
    else:
        file_id = update.message.photo[-1]['file_id']
    file = bot.getFile(file_id)
    image = file.download_as_bytearray()
    return image, file_id


def draw_contours(image_array, metadata):
    """ Функция отрисовки контура и подсчета кол-во особей
    Обрабатываем результат работы моделей, извлекая полученный класс животного
    Args:
        image_array: arr - массив-представление изображения
        metadata: json - словарь, содержащий ответ работы моделей
    Return:
        counter_dict: dict - словарь с кол-вом определенных животных
    """
    
    for bbox in tqdm(metadata["bbox"]):
        # class_name = bbox['class_name']

        # confidence = bbox['confidence']

        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

        center_coords = int((botRightCorner[0] + topLeftCorner[0]) / 2), int((botRightCorner[1] + topLeftCorner[1]) / 2)

        cv2.putText(image_array, '*', 
                            center_coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.3, (255, 0, 0),
                            1,
                            1)

    return len(metadata["bbox"])