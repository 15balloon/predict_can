#%%
from imgaug.augmenters.arithmetic import Dropout
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#%%
location = "./"
class_name = ["칠성사이다", "코카콜라", "데미소다 애플", "데미소다 자몽", "데미소다 오렌지", "환타",
            "핫식스", "레쓰비", "립톤 아이스티", "맥콜", "마운틴듀", "갈아만든 배",
            "펩시", "포카리스웨트", "솔의눈", "스프라이트", "토레타", "웰치스"]

#%%
# Data
X_train, X_test, y_train, y_test = np.load(location + "image_data.npy", allow_pickle=True)

#%%
# print shape & image
print(X_train.shape)
print(y_train.shape)

plt.figure()
plt.imshow(X_train[0])
plt.show()

#%%
# CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 7, activation=tf.keras.activations.relu, padding="same", input_shape=X_train[0].shape),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(256, 3, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.Conv2D(256, 3, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(18, activation=tf.keras.activations.softmax)
])

#%%
# Take model summary
tf.keras.utils.plot_model(model, to_file=location+'model_summary.png', show_shapes=True, show_layer_names=True)

#%%
# Compiling model
model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy,
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
metrics=[tf.keras.metrics.sparse_categorical_accuracy])

# %%
# Load model
model = tf.keras.models.load_model(location + 'drink_model.h5')

# %%
# Load best model
model = tf.keras.models.load_model(location + 'best_model.h5')

#%%
# Learning model 1 - callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
mc = tf.keras.callbacks.ModelCheckpoint(location + 'best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), shuffle=True, callbacks=[early_stopping, mc])
# history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), shuffle=True, callbacks=[mc])

#%%
# Learning model 2
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), shuffle=True)

#%%
# Save weight
model.save(location + 'drink_model.h5')

#%%
# Save history
np.save(location + 'history.npy',history.history)

#%%
# Load history
history = np.load(location + 'history.npy', allow_pickle='TRUE').item()

#%%
# Evaluate
model.evaluate(X_test, y_test)

#%%
# Plot history - learning
def plot_history(history, key='sparse_categorical_accuracy'):
    # acc & loss
    fig1 = plt.figure(figsize=(16,10))
    plt.plot(history.epoch, history.history['val_'+key], 'b--', label='Val')
    plt.plot(history.epoch, history.history[key], 'b-', label='Train')
    plt.plot(history.epoch, history.history['val_loss'], 'r--', label='Val')
    plt.plot(history.epoch, history.history['loss'], 'r-', label='Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title() + ' / loss')
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.ylim([0, 1])

    return fig1

fig1 = plot_history(history)

#%%
# Plot history - load
def plot_history(history, key='sparse_categorical_accuracy'):
    fig1 = plt.figure(figsize=(16,10))
    epc = 100
    plt.plot(range(epc), history['val_'+key], 'b--', label=' Val')
    plt.plot(range(epc), history[key], 'b-', label=' Train')
    plt.plot(range(epc), history['val_loss'], 'r--', label='Loss Val')
    plt.plot(range(epc), history['loss'], 'r-', label='Loss Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title() + ' / loss')
    plt.legend()

    plt.xlim([0,epc])
    plt.ylim([0,1])

    return fig1

fig1 = plot_history(history)

#%%
# Save plot history
fig1.savefig(location + 'acc_loss.png')

# %%
# image preprocessing for predict
imgs = []
loc_img = location + "new_test\\"
filelist = os.listdir(loc_img)
for i in range(len(filelist)):
    img = tf.keras.preprocessing.image.load_img(loc_img + filelist[i], target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_tf = img_array / 255.
    imgs.append(img_tf)

imgs = np.array(imgs)

#%%
# Predict image
X_pred = imgs
y_pred = model.predict_classes(X_pred)
y_proba = model.predict(X_pred)
print(y_proba.round(2))

#%%
# Predict test set
xstart = 10
step = 2
X_pred = X_test[xstart:50:step]
y_pred = model.predict_classes(X_pred)
y_proba = model.predict(X_pred)
print(y_proba.round(2))

#%%
# Convert idx2name
pred_name = np.array(class_name)[y_pred]
print(pred_name)

#%%
# Print result
import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname="C:\\Windows\\Fonts\\malgun.ttf", size=8)

fig = plt.figure(figsize=(10,16))
for i in range(len(X_pred)):
    plt.subplot((len(X_pred)//5 + 1), 5, i+1)
    # plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_pred[i])
    # plt.xlabel('{}\nGT:{}'.format(pred_name[i], class_name[y_test[xstart+i*step]]), fontproperties=fontprop)
    plt.xlabel('{}'.format(pred_name[i]), fontproperties=fontprop)
plt.show()

#%%
# Save result
fig.savefig(location + 'predict_new_test.png')