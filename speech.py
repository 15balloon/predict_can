import numpy as np
import tensorflow as tf
from gtts import gTTS
import playsound
import os

location = "./"
class_name = ["칠성사이다", "코카콜라", "데미소다 애플", "데미소다 자몽", "데미소다 오렌지", "환타",
            "핫식스", "레쓰비", "립톤 아이스티", "맥콜", "마운틴듀", "갈아만든 배",
            "펩시", "포카리스웨트", "솔의눈", "스프라이트", "토레타", "웰치스"]


def speech(path):
    drink_name = load_cnn(path)
    tts(drink_name)
    return drink_name

def load_cnn(path):
    model = tf.keras.models.load_model(location + 'drink_model.h5')
    print(path)
    img = tf.keras.preprocessing.image.load_img(path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_tf = np.array(img_array / 255.)
    X_pred = tf.expand_dims(img_tf, axis=0)
    y_pred = model.predict_classes(X_pred)
    pred_name = np.array(class_name)[y_pred]
    return pred_name[0]

def tts(name):
    tts = gTTS(text=name, lang='ko')
    tts.save(location + 'drink_name.mp3')
    playsound.playsound(location + 'drink_name.mp3')
    os.remove(location + 'drink_name.mp3')