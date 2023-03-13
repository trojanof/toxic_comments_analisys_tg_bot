import time
import re
import string
import logging
from tg_settings import TG_TOKEN
from aiogram import Bot, Dispatcher, executor, types
import random
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))

API_TOKEN = TG_TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

with open('models/mod_vectorizer.pkl', 'rb') as fr:
    vec_from_disk = pickle.load(fr)
    
with open('models/model.pkl', 'rb') as fr:
    mdl_from_disk = pickle.load(fr)

def data_preprocessing(text: str) -> str:

    text = text.lower()
    text = re.sub('<.*?>', '', text) # html tags
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words] 
    text = ' '.join(text)
    return text    

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.username
    user_id = message.from_user.id
    text = f'''Привет ,{user_name}, данный бот создан для оценки уровня токсичности переданного сообщения, обученная модель логистической регрессии анализирует текст и возвращает оценку токсичности текста в процентах.'''
    logging.info(f'{user_name=}{user_id=} sent message:{message.text}')
    await message.reply(text)


@dp.message_handler()
async def toxic_level(message: types.Message):
    user_name = message.from_user.username
    user_id = message.from_user.id
    text = message.text
    text = data_preprocessing(text)
    msg_vector = vec_from_disk.transform([text]) # message_text - сообщение пользователя в виде строки (str)
    msg_df = pd.DataFrame.sparse.from_spmatrix(msg_vector)
    msg_df.columns = vec_from_disk.get_feature_names_out()
    probability = mdl_from_disk.predict_proba(msg_df)[0][1]
    if probability > 0.5:
        result = f'Сообщение скорее всего токсичное, вероятность {probability*100:.1f} %'
    else:
        result = f'Сообщение скорее всего не токсичное, вероятность {(1-probability)*100:.1f} %'
    logging.info(f'{user_name=}, {user_id=} sent message: {text=} bot_return: {result=}')
    await bot.send_message(user_id, result)


if __name__ == '__main__':
    executor.start_polling(dp)