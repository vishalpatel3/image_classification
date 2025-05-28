import os
from PIL import Image
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def save_images(images, labels, folder):
    for i, (img, label) in enumerate(zip(images, labels)):
        class_dir = os.path.join(folder, class_names[label[0]])
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f'{i}.png')
        Image.fromarray(img).save(img_path)

save_images(x_train, y_train, 'data/train')
save_images(x_test, y_test, 'data/test')
