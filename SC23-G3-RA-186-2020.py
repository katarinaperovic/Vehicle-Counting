import os
import sys
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')


def sliding_window_large(image, stepSize, windowSize):
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            window = image[y:y + windowSize[1], x:x + windowSize[0]]
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue
            
            hog_features = hog.compute(window)
            prediction = clf_svm.predict(hog_features.reshape(1,-1))
            probability = clf_svm.predict_proba(hog_features.reshape(1, -1))[0][1]  
            
            if prediction == 1 and probability > 0.91:  
                yield (x, y, x + windowSize[0], y + windowSize[1])
            

def non_max_suppression(detections, threshold=0.5):
    if len(detections) == 0:
        return []

    picked_detections = []

    while len(detections) > 0:
        
        picked_detections.append(detections[0])

        detections = detections[1:]
        new_detections = []

        for det in detections:
           
            x1, y1, _, _ = picked_detections[-1]  
            x2, y2, _, _ = det 

            overlap = calculate_overlap((x1, y1, x1 + winW, y1 + winH), (x2, y2, x2 + winW, y2 + winH))

            if overlap <= threshold:
                new_detections.append(det)

        
        detections = new_detections

    return picked_detections

def calculate_overlap(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    overlap = inter_area / float(boxA_area + boxB_area - inter_area)

    return overlap

def detect_line(img):
    
    edges_img = cv2.Canny(img, 50, 110, apertureSize=3)
    plt.imshow(edges_img, "gray")
    
    min_line_length = 400
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
 
    x1 = lines[0][0][0]
    y1 = img.shape[0] - lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = img.shape[0] - lines[0][0][3]
    
    return (x1, y1, x2, y2)

def get_line_params(line_coords):
    k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
    n = k * (float(-line_coords[0])) + float(line_coords[1])
    return k, n

def detect_cross(x, y, k, n):
    yy = k * x + n
    
    return -17 <= (yy - y) <= 22

directory=sys.argv[1]
videos_folder = os.path.join('videos')

train_dir = os.path.join('pictures')

pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_name)
    img = load_image(img_path)
    
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)
        

nbins = 9 
cell_size = (16, 16) 
block_size = (3, 3) 

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)


pos_features = []
neg_features = []
labels = [] 

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf_svm = SVC(kernel='linear', probability=True) 
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)

predictions = []

for filename in os.listdir(videos_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(videos_folder, filename)

            sum_of_cars = 0
            k = 0
            n = 0
            frame_num = 1
            cap = cv2.VideoCapture(video_path)
                
            video_count = pd.read_csv('counts.csv')[pd.read_csv('counts.csv')['Naziv_videa'] == filename]['Broj_prelaza'].values[0]
            frame_skip = 5
            while True:
                    #frame_num += 1
                    cap.set(1, frame_num)
                    grabbed, frame = cap.read()

                    if not grabbed or frame.size == 0:
                        break
                    
                    frame = cv2.resize(frame, (1919, 1079))
                    fg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    cropped_frame = fg[280:800, 750:1250]
                        
                    if frame_num == 1:
                        line_coords = detect_line(cropped_frame)
                        k, n = get_line_params(line_coords)
                        
                        line_left_x = line_coords[0]
                        line_right_x = line_coords[2]
                
                    winW, winH = (60, 120)
                    stepSize = 8
                
                    detections = []
                    for (x, y, x2, y2) in sliding_window_large(cropped_frame, stepSize=stepSize, windowSize=(winW, winH)):
                        detections.append((x, y, x2, y2))

                    final_detections = non_max_suppression(detections, threshold=0.5)

                    for (x1, y1, x2, y2) in final_detections:
            
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
            
                        if detect_cross(center_x, center_y, k, n):
                            sum_of_cars += 1  
                            
                    frame_num += frame_skip
                    
                    
                    
            cap.release()
            predictions.append(sum_of_cars)
            
            print(f"{filename}-{video_count}-{sum_of_cars}")


true_values = pd.read_csv('counts.csv', usecols=['Broj_prelaza'])['Broj_prelaza'].values

print(mean_absolute_error(true_values,predictions))



