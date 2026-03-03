import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
from collections import Counter
n_feature =8
def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        threshold = 4
        binary = gray > threshold
    lb = label(binary)
    props = regionprops(lb)
    props = sorted(props, key=lambda x: x.bbox[1])
    p=props[0]
    return np.array([p.eccentricity,4 * p.euler_number,p.extent, p.solidity,(p.axis_major_length / p.axis_minor_length), p.extent, p.centroid_local[0]/p.image.shape[0], p.centroid_local[1]/p.image.shape[1]], dtype='f4')
#/p.image.shape[0]
def make_train(path):
    train = []
    responses = []
    class_names = []
    ncls = 0
    for folder in path.iterdir():
        ncls += 1
        #print(folder.name, ncls)
        
        if folder.is_dir():
            #print(f"Обрабатываем папку: {folder}")
            class_names.append(folder.name[-1])
            for img_path in folder.glob("*.png"):
                train.append(extractor(imread(img_path)))
                responses.append(ncls)
                

                
    
    train = np.array(train, dtype="f4").reshape(-1, n_feature)
    responses = np.array(responses, dtype='f4').reshape(-1, 1)
    return train, responses, class_names

def merge_props(props):
    props = sorted(props, key=lambda x: x.bbox[1])
    
    merged = []
    used = [False] * len(props)
    
    for i, prop1 in enumerate(props):
        if used[i]:
            continue
            
        y1_1, x1_1, y2_1, x2_1 = prop1.bbox
        cy1, cx1 = prop1.centroid
        
        min_y, min_x = y1_1, x1_1
        max_y, max_x = y2_1, x2_1
        
        used[i] = True
        
        for j, prop2 in enumerate(props):
            if not used[j]:
                y1_2, x1_2, y2_2, x2_2 = prop2.bbox
                cy2, cx2 = prop2.centroid
                
                if (min_x <= cx2 <= max_x):
                    min_y = min(min_y, y1_2)
                    min_x = min(min_x, x1_2)
                    max_y = max(max_y, y2_2)
                    max_x = max(max_x, x2_2)
                    used[j] = True
        merged.append((min_y, min_x, max_y, max_x))
    
    return merged

with open('../path.txt', 'r', encoding='utf-8') as file:
    root_path = Path(file.read())

dir_path = root_path / "knn_ocr"
task_path = dir_path / "task"
train_path = task_path / "train"
for i in range(7):
    image = imread(task_path / f"{i}.png")
    train, responses, class_names = make_train(train_path)
    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)
    gray = image.mean(2)
    threshold = 4
    binary = gray > threshold 
    lb = label(binary)
    props = regionprops(lb)
    merged_boxes = merge_props(props)

    find = []
    for bbox in merged_boxes:
        y1, x1, y2, x2 = bbox
        img = binary[y1:y2, x1:x2]
        find.append(extractor(img))

    find = np.array(find, dtype="f4").reshape(-1, n_feature)
    ret, results, neighbours, dist = knn.findNearest(find, n_feature)
    results_flat = results.flatten()
    results_names = [class_names[int(r)-1] for r in results_flat if r > 0]
    neighbours_names = [[class_names[int(n)-1] for n in row if n > 0] for row in neighbours]


    threshold_bbox = 30

    x2_p = None
    text_with_spaces = ""

    for bbox, char in zip(merged_boxes, results_names):
        y1, x1, y2, x2 = bbox
        
        if x2_p is not None:
            distance = x1 - x2_p
            if distance > threshold_bbox:
                text_with_spaces += " "
        
        text_with_spaces += char
        x2_p = x2

    print(text_with_spaces)
