import telegram
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from telegram import InputMediaPhoto
from telegram.ext import Updater
# from credentials import bot_token
import cv2
import requests
import os 
import glob

from bot_commands import start, cut_video
from utils import process_image, transform_pil_image_to_bytes, get_video, get_image_bytes, draw_contours


TOKEN = os.getenv('BOT_TOKEN')
URL = 'http://localhost:5008/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('initializing completed')

def bot_image_processing(bot, update):
    """Функция обработки запроса с изображением
    
    Получаем и обрабатываем запрос с бэка, формируем сообщение в бота

    Args:
        bot: bot - тело нашего бота
        update: command - параметр для обновления данных в боте
    """
    print("Получение изображения")
    image_bytes, _ = get_image_bytes(bot, update)

    image_array = process_image(image_bytes)
    _, img_encoded = cv2.imencode('.jpg', image_array)
    data = img_encoded.tostring()
    print("Изображение получено, отправляю запрос на распознавание")
    response = requests.post(URL, data=data, headers=headers) # отправляем запрос на бэк
    metadata = response.json()['image'] # обрабатываем полученный ответ с результатами работы модели
    print("Отрисовка контуров")
    walrus_counter = draw_contours(image_array, metadata)
    image = transform_pil_image_to_bytes(image_array)

    print('Изображение обработано')
    text = f"На фото мы смогли распознать {walrus_counter} ненецкого(их) льва(ов)!!😎😎😎"
    bot.send_photo(chat_id=update.message.chat_id, photo=image)
    bot.send_message(chat_id=update.message.chat_id, text=text,
                    reply_to_message_id=update.message.message_id,
                    parse_mode=telegram.ParseMode.HTML)





start_handler = CommandHandler(['start', 'help'], start)
cut_video_handler = CommandHandler(['cut_video'], cut_video)

process_image_handler = MessageHandler(Filters.photo | Filters.document, bot_image_processing)
# process_video_handler =  MessageHandler(Filters.video, bot_video_preprocessing)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(cut_video_handler)
dispatcher.add_handler(process_image_handler)
# dispatcher.add_handler(process_video_handler)
updater.start_polling()