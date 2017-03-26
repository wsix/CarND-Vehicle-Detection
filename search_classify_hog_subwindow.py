import os
import glob
import cv2
import time
import numpy as np
import matplotlib.image as mpimg
from datetime import datetime
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from subwindow_function import *

orient = 10  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 64  # Number of histogram bins
ystart = 400  # Min and max in y to search in slide_window()
ystop = 700  # Min and max in y to search in slide_window()
scale = 1

cars = glob.glob('data/vehicles/*/*.png')
notcars = glob.glob('data/non-vehicles/*/*.png')
# Extract features: HOG features, spatially binned color and histograms of color
car_features = extract_features(cars, spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block)
notcar_features = extract_features(notcars, spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
if os.path.exists('classifier/vehicleClf.pkl'):
    clf = joblib.load("classifier/vehicleClf.pkl")
else:
    parameters = {'C': [1, 10, 100]}
    svc = LinearSVC()
    clf = GridSearchCV(svc, parameters)
    # Check the training time for the SVC
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    joblib.dump(clf, 'classifier/vehicleClf.pkl')

t = time.time()

cap = cv2.VideoCapture('project_video.mp4')
count = 0
while(cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    (r, g, b) = cv2.split(image)
    image = cv2.merge([b, g, r])
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # use different scale and shape of slide windows to find cars
    box_list = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, (64, 32))
    box_list_ms = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, (48, 48))
    box_list_hs = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, (32, 64))
    box_list_ss = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, (16, 16))
    box_list.extend(box_list_ms)
    box_list.extend(box_list_hs)
    box_list.extend(box_list_ss)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 4)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    # image_filename = os.path.join('output_images/test_output/', timestamp)
    image_filename = os.path.join('output_images/project_output/', timestamp)
    mpimg.imsave('{}.jpg'.format(image_filename), draw_img)
    if count % 20 == 0:
        print('{} Done!'.format(count))
    count += 1
print('Complete!')
