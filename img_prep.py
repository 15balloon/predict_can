from sklearn.model_selection import train_test_split
import numpy as np 
import tensorflow as tf
import os


# image load
def read_images(location):
    X = []
    Y = []
    categories = ["chilsungcider", "cocacola", "demisoda-apple", "demisoda-grapefruit", "demisoda-orange", "fanta",
                 "hotsix", "letsbe", "liptonicetea", "mccol", "mountaindew", "pear",
                  "pepsi", "pocarisweat", "sol", "sprite", "toreta", "welchs"]
    for idx, cate in enumerate(categories):
        label = idx
        img_path_base = location + cate + "\\"
        filelist = os.listdir(img_path_base)
        for i in range(len(filelist)):
            img_path = img_path_base + filelist[i]
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
            img_tf = tf.keras.preprocessing.image.img_to_array(img)
            # img_tf.shape (64, 64, 3)
            data = img_tf / 255.
            X.append(data)
            Y.append(label)
    return X, Y


def main():
    # image load
    location = "./image\\"

    X, Y = read_images(location)
    X = np.array(X)
    Y = np.array(Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=39)
    XY = (X_train, X_test, y_train, y_test)
    save_loca = location[:-6] + "image_data.npy"
    np.save(save_loca, XY)
    print("save complete: {0}image_data.npy".format(location[:-6]))
    # X_train, X_test, y_train, y_test = np.load("image_data.npy")


if __name__ == "__main__":
    main()