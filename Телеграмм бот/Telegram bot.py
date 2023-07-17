#!/usr/bin/env python
# coding: utf-8

"""
Модуль для обучения и использования модели Word2Vec для предсказания названий серий по синопсисам.
"""

import re
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial
import telebot
from telebot import types


# Устанавливаем параметры для отображения данных в pandas
pd.set_option("display.max_columns", None)

# Скачиваем необходимые ресурсы для nltk
nltk.download('stopwords')
nltk.download('punkt')

# Создаем объект переводчика с помощью Google Translate API
TRANSLATOR = Translator(service_urls=['translate.google.com'])


class ModelWord2Vec:
    """
    Класс для обучения и использования модели Word2Vec.

    Аргументы:
        final_df (pandas.DataFrame): pandas DataFrame, содержащий данные для обучения модели.

    Атрибуты:
        vectors (list): Список векторов, представляющих слова в DataFrame final_df.
        model (gensim.models.Word2Vec): Обученная модель Word2Vec.

    Методы:
        create_model(self): Обучает модель Word2Vec.
        vec_df(self): Преобразует DataFrame final_df в DataFrame, содержащий векторы для каждого слова.
        predict(self, synopsis: str, df: pd.DataFrame): Предсказывает название серии для данного синопсиса.
    """

    def __init__(self, final_df):
        self.vectors = []
        self.final_df = final_df
        self.model = None

    def create_model(self):
        """
        Обучает модель Word2Vec.

        Возвращает:
            gensim.models.Word2Vec: Обученная модель Word2Vec.
        """
        self.model = Word2Vec(
            self.final_df["synopsis_tokens"],
            window=6,
            vector_size=50,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            sg=0
        )
        self.model.train(
            self.final_df["synopsis_tokens"],
            total_examples=len(self.final_df["synopsis_tokens"]),
            epochs=1
        )
        return self.model

    def vec_df(self):
        """
        Преобразует DataFrame final_df в DataFrame, содержащий векторы для каждого слова.

        Возвращает:
            pandas.DataFrame: DataFrame, содержащий векторы для каждого слова.
        """
        for synopsis_tokens in self.final_df["synopsis_tokens"]:
            synopsis_vector = np.zeros(self.model.vector_size)
            num_tokens = 0
            for token in synopsis_tokens:
                if token in self.model.wv.key_to_index:
                    vector = self.model.wv.get_vector(token)
                    synopsis_vector += vector
                    num_tokens += 1
            if num_tokens > 0:
                synopsis_vector /= num_tokens
                self.vectors.append(synopsis_vector)
        self.final_df["vectors"] = self.vectors

        return self.final_df

    def predict(self, synopsis: str, df: pd.DataFrame) -> pd.Series:
        """
        Предсказывает название серии для данного синопсиса.

        Аргументы:
            synopsis (str): Синопсис серии.
            df (pd.DataFrame): DataFrame, содержащий векторы и названия серий.

        Возвращает:
            pandas.Series: Предсказанное название серии и соответствующий вектор.
        """
        # Предобрабатываем синопсис
        synopsis = preprocessor_text(synopsis)

        # Получаем список слов в синопсисе, которые есть в словаре модели
        lst = set([i for i in synopsis.split() if i in self.model.wv.key_to_index])

        # Получаем векторы слов в синопсисе
        context_word_vectors = [self.model.wv[word] for word in lst]

        # Рассчитываем средний вектор слов для синопсиса
        predicted_vector = np.mean(context_word_vectors, axis=0)

        # Рассчитываем косинусное сходство между предсказанным вектором и векторами в наших данных
        similarities = df['vectors'].map(lambda x: 1 - spatial.distance.cosine(
            np.array(object=x, dtype='float16'),
            predicted_vector.astype('float16')
                )
            )

        # Возвращаем название серии с наибольшим сходством
        return df.iloc[similarities.argmax()]



def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Аргументы:
        df: Фрейм данных.

    Возвращает:
        df + токенизированные колонки.
    """
    df.dropna(subset=['release_year', 'rating', 'cast'], inplace=True)
    df['release_year'] = df['release_year'].astype(int)
    df["synopsis_tokens"] = df["synopsis1"].apply(word_tokenize)
#     df['series_title'] = df['series_title'].drop_duplicates().dropna()

    return df




def preprocessor_text(text: str) -> str:
    """
    Нормализует текстовую строку.

    Аргументы:
        text: Текстовая строка для нормализации.

    Возвращает:
        Нормализованную текстовую строку.
    """
    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')

    tokens = word_tokenize(text)

    # удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    processed_text = ' '.join(filtered_tokens)

    return processed_text


bot = telebot.TeleBot('6264185691:AAF7DgTJ0CLAOnMqgGKsE2NgW7g9DqwiSFs')

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(
        message.chat.id,
        "Привет, {0.first_name}! Я бот системы рекомендаций сериалов. Введите описание, и я дам вам рекомендацию.".format(message.from_user)
    )
@bot.message_handler(commands=['menu'])   
def show_main_menu(message):
    global keyboard
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = types.KeyboardButton("✒️ Повторно ввести описание")
    button2 = types.KeyboardButton("ℹ️ Дополнительные сведения о сериале")
    markup.add(button1, button2)
    bot.send_message(message.chat.id, text="Меню:", reply_markup=markup)
    
    
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    global predict
    df = pd.read_csv(
        "C:/Users/fourz/OneDrive/Рабочий стол/Word2Vec/filter_data.csv",
        usecols=lambda column: column != 'Unnamed: 0' and column != 'synopsis_tokens'
    )
    synaps = message.text
    
    if synaps == "/start":  
        bot.send_message(
            message.chat.id,
            "Привет, {0.first_name}! Я бот системы рекомендаций сериалов. Введите описание, и я дам вам рекомендацию.".format(message.from_user)
        )
        return  

    if message.content_type == 'text' and (
        message.text.startswith("✒️") or
        message.text.startswith("ℹ️") or
        message.text.endswith('🎭') or 
        message.text.endswith('⭐️') or
        message.text.endswith('📅') or
        message.text.endswith('📝') or
        message.text.endswith('🎬') or
        message.text.endswith('🌐')
    ):
        return 

    final_df = preprocess(df)

    pp = preprocessor_text(synaps)

    model_word2vec = ModelWord2Vec(final_df)
    model = model_word2vec.create_model()

    emb = model_word2vec.vec_df()

    pred = model_word2vec.predict(pp, final_df)

    predict = pred[['series_title', 'release_year', 'runtime', 'genre', 'rating', 'cast', 'synopsis', 'end_year']]
    
    bot.send_message(message.chat.id, f"Рекомендация: {predict['series_title']}")


@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    
    handle_message(message)
    if message.text.startswith("✒️ Повторно ввести описание"):
        bot.send_message(message.chat.id, 'Введите текст:', reply_markup=keyboard)
        handle_message(message)
#         show_main_menu(message)

    elif message.text.startswith("ℹ️ Дополнительные сведения о сериале"):
        bot.send_message(message.chat.id, message.text)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
        btn1 = types.KeyboardButton("Узнать актерский состав 🎭")
        btn2 = types.KeyboardButton("Узнать рейтинг сериала ⭐️")
        btn3 = types.KeyboardButton("Узнать годы выхода сериала 📅")
        btn4 = types.KeyboardButton("Узнать описание 📝")
        btn5 = types.KeyboardButton("Узнать жанр 🎬")
        back = types.KeyboardButton("Получить ссылку на фильм 🌐")
        markup.add(back)
        markup.row(btn2, btn3, btn4)
        markup.add(btn5, btn1)
        bot.send_message(message.chat.id, text="Выберите чего хотите узнать:", reply_markup=markup)

        
    elif message.text.startswith("Узнать актерский состав 🎭"):
        bot.send_message(message.chat.id, f"Актеры: {predict['cast']}")
        show_main_menu(message)

    elif message.text.startswith("Узнать рейтинг сериала ⭐️"):
        bot.send_message(message.chat.id, f"Рейтинг сериала: {predict['rating']}")
        show_main_menu(message)

    elif message.text.startswith("Узнать годы выхода сериала 📅"):
        bot.send_message(message.chat.id, f"Годы выхода: {predict['release_year']}")
        show_main_menu(message)

    elif message.text.startswith("Узнать описание 📝"):
        bot.send_message(message.chat.id, f"Описание: {predict['synopsis']}")
        show_main_menu(message)

    elif message.text.startswith("Узнать жанр 🎬"):
        bot.send_message(message.chat.id, f"Жанр: {predict['genre']}")
        show_main_menu(message)

    elif message.text.startswith("Получить ссылку на фильм 🌐"):
        bot.send_message(message.chat.id,"https://www.imdb.com/title/tt0944947/plotsummary/")
        show_main_menu(message)

    else:
        bot.send_message(message.chat.id, text="На такую команду я не запрограммирован..")
        show_main_menu(message)



@bot.message_handler(func=lambda message: True)
def handle_other_messages(message):
    pass  


bot.polling(none_stop=True)