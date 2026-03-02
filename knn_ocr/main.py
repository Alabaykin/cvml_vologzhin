import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
from collections import Counter

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        threshold = 255
        binary = gray < threshold
    lb = label(binary)
    props = regionprops(lb)
    return np.array([props[0].eccentricity, (props[0].area / np.pi) ** 0.5], dtype='f4')

def make_train(path):
    train = []
    responses = []
    class_names = []
    ncls = 0
    for folder in path.iterdir():
        ncls += 1
        print(folder.name, ncls)
        if folder.is_dir():
            print(f"Обрабатываем папку: {folder}")
        
            for img_path in folder.glob("*.png"):
                train.append(extractor(imread(img_path)))
                responses.append(ncls)
                class_names.append(folder.name[-1])

                
    
    train = np.array(train, dtype="f4").reshape(-1, 2)
    responses = np.array(responses, dtype='f4').reshape(-1, 1)
    return train, responses, class_names

with open('../path.txt', 'r', encoding='utf-8') as file:
    root_path = Path(file.read())

dir_path = root_path / "knn_ocr"
task_path = dir_path / "task"
train_path = task_path / "train"

image = imread(task_path / "1.png")
train, responses, class_names = make_train(train_path)
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)
gray = image.mean(2)
binary = gray

threshold = 50 
binary = gray > threshold 
lb = label(binary)
props = regionprops(lb)
find = []
for prop in props:
    find.append(extractor(prop.image))
find = np.array(find, dtype="f4").reshape(-1, 2)

ret, results, neighbours, dist = knn.findNearest(find, 5)
results_flat = results.flatten()
results_names = [class_names[int(r)-1] for r in results_flat if r > 0]
neighbours_names = [[class_names[int(n)-1] for n in row if n > 0] for row in neighbours]

print("Числовые результаты:", results_flat)
print("Названия классов:", results_names)
print("Соседи (названия):", neighbours_names)
print("Статистика:", Counter(results_names))
plt.imshow(image)
plt.show()