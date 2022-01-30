import numpy as np
import os, cv2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# Reading the images
train_cats = np.zeros((5153, 128, 128), dtype="uint8")
train_dogs = np.zeros((4739, 128, 128), dtype="uint8")
train_wild = np.zeros((4738, 128, 128), dtype="uint8")
val_cats = np.zeros((500, 128, 128), dtype="uint8")
val_dogs = np.zeros((500, 128, 128), dtype="uint8")
val_wild = np.zeros((500, 128, 128), dtype="uint8")

train_cat_index = 0
train_dog_index = 0
train_wild_index = 0

val_cat_index = 0
val_dog_index = 0
val_wild_index = 0

for root, dirs, files in os.walk("."):
    if "train" in root:
        if "cat" in root:
            for file in files:
                train_cats[train_cat_index, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file, 0)),
                                                                  dsize=(128, 128))
                train_cat_index += 1

        elif "dog" in root:
            for file in files:
                train_dogs[train_dog_index, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file, 0)),
                                                                  dsize=(128, 128))
                train_dog_index += 1

        elif "wild" in root:
            for file in files:
                train_wild[train_wild_index, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file, 0)),
                                                                   dsize=(128, 128))
                train_wild_index += 1

    elif "val" in root:
        if "cat" in root:
            for file in files:
                val_cats[val_cat_index, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file, 0)),
                                                              dsize=(128, 128))
                val_cat_index += 1

        elif "dog" in root:
            for file in files:
                val_dogs[val_dog_index, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file, 0)),
                                                              dsize=(128, 128))
                val_dog_index += 1

        elif "wild" in root:
            for file in files:
                val_wild[val_wild_index, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file, 0)),
                                                               dsize=(128, 128))
                val_wild_index += 1

types =["Cat", "Dog", "Wild Animal"] 

# Store original images
orig_val_cats = val_cats
orig_val_dogs = val_dogs
orig_val_wild = val_wild

# Make it 0-1
train_cats = train_cats / 255
train_dogs = train_dogs / 255
train_wild = train_wild / 255

val_cats = val_cats / 255
val_dogs = val_dogs / 255
val_wild = val_wild / 255

# Total Train Data
total_train = train_cats
total_train = np.append(total_train, train_dogs, axis=0)
total_train = np.append(total_train, train_wild, axis=0)

train_lbls = np.zeros(5153 + 4739 + 4738)
train_lbls[5153: 5153 + 4739] = 1
train_lbls[5153 + 4739: 5153 + 4739 + 4738] = 2

# Total Test Data
total_test = val_cats
total_test = np.append(total_test, val_dogs, axis=0)
total_test = np.append(total_test, val_wild, axis=0)

test_lbls = np.zeros(500 * 3)
test_lbls[500: 1000] = 1
test_lbls[1000: 1500] = 2

# Store original images
orig_total_test = orig_val_cats
orig_total_test = np.append(orig_total_test, orig_val_dogs, axis=0)
orig_total_test = np.append(orig_total_test, orig_val_wild, axis=0)

# Flatting the images for the SVM
train_flattened = total_train.reshape(len(train_lbls), -1)
test_flattened = total_test.reshape(len(test_lbls), -1)

# Finding the PCA for the data to make it bearable
pca = PCA(n_components=100)
pca.fit(train_flattened)

# Transforming the data with pca
train_flattened_pca = pca.transform(train_flattened)
test_flattened_pca = pca.transform(test_flattened)

# Loading the best model produced
clf = joblib.load("bestSVM.sav")

print("The performance evaluation metrics ")
predictions = clf.predict(test_flattened_pca)

print("The classification report of the model")
print(classification_report(test_lbls, predictions, target_names=types))

print("The confusion matrix for the model")
print(confusion_matrix(test_lbls, predictions, labels=range(3)))



