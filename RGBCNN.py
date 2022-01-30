import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import models, layers, regularizers
import pandas as pd
import numpy as np
import os, cv2
import tensorflow as tf
import time
import optuna
from tensorflow import keras
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



def init_Model():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(128, 128, 3)))
    model.add(layers.Conv2D(16, (3, 3), activation='leaky_relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, (3, 3), activation='leaky_relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(40, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    return model

def objective(trial):
    model = init_Model()
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
                  )

    # Train the Model
    history = model.fit(total_train, train_lbls, shuffle=True, epochs=50, batch_size=batch_size, validation_split=0.20, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath='best.hdf5',
                                           monitor='val_loss',
                                           save_best_only=True,
                                           mode='min')])

    array = np.array(history.history['val_loss'])
    return array.min()


# Reading the images
train_cats = np.zeros((5153, 128, 128, 3), dtype="uint8")
train_dogs = np.zeros((4739, 128, 128, 3), dtype="uint8")
train_wild = np.zeros((4738, 128, 128, 3), dtype="uint8")
val_cats = np.zeros((500, 128, 128, 3), dtype="uint8")
val_dogs = np.zeros((500, 128, 128, 3), dtype="uint8")
val_wild = np.zeros((500, 128, 128, 3), dtype="uint8")

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
                train_cats[train_cat_index, :, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file)),
                                                                  dsize=(128, 128))
                train_cat_index += 1

        elif "dog" in root:
            for file in files:
                train_dogs[train_dog_index, :, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file)),
                                                                  dsize=(128, 128))
                train_dog_index += 1

        elif "wild" in root:
            for file in files:
                train_wild[train_wild_index, :, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file)),
                                                                   dsize=(128, 128))
                train_wild_index += 1

    elif "val" in root:
        if "cat" in root:
            for file in files:
                val_cats[val_cat_index, :, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file)),
                                                              dsize=(128, 128))
                val_cat_index += 1

        elif "dog" in root:
            for file in files:
                val_dogs[val_dog_index, :, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file)),
                                                              dsize=(128, 128))
                val_dog_index += 1

        elif "wild" in root:
            for file in files:
                val_wild[val_wild_index, :, :, :] = cv2.resize((cv2.imread(os.getcwd() + root[1:] + "\\" + file)),
                                                               dsize=(128, 128))
                val_wild_index += 1

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

[tot_train_dim1, tot_train_dim2, tot_train_dim3, tot_train_dim4] = total_train.shape
[tot_test_dim1, tot_test_dim2, tot_test_dim3, tot_test_dim4] = total_test.shape
indexes = np.random.choice(tot_train_dim1, tot_train_dim1, replace=False)
total_train = total_train[indexes]
train_lbls = train_lbls[indexes]

indexes = np.random.choice(tot_test_dim1, tot_test_dim1, replace=False)
total_test = total_test[indexes]
test_lbls = test_lbls[indexes]

# True for creating new Model, False for using existing one
if True:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    start_time = time.time()
    print("Best trial:")
    trial = study.best_params
    print(trial)
    lr = trial['lr']
    batch_size = trial['batch_size']
    model = init_Model()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
                  )

    # Train the Model
    history = model.fit(total_train, train_lbls, shuffle=True, epochs=50, batch_size=batch_size, validation_split=0.20, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath='best.hdf5',
                                           monitor='val_loss',
                                           save_best_only=True,
                                           mode='min')])
    end_time = time.time()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.legend(["Validation Loss", "Train Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("RGB CNN Learning Curve")
    plt.show()
    print()
    print("Training Duration: " + str(end_time - start_time))


model = tf.keras.models.load_model('best.hdf5')

# Evaluate the Accuracy
print()
print()
print("Testing on the test data")
model.evaluate(total_test, test_lbls)

predictions = []
for k in total_test:
    predictions.append(np.argmax(model.predict(np.reshape(k, (1, 128, 128, 3)))))

predictions = np.array(predictions)
predictions = np.reshape(predictions, test_lbls.shape)
res = tf.math.confusion_matrix(test_lbls, predictions, num_classes=3)
cat = res[0]
dog = res[1]
wild = res[2]
# Printing the result
print('Confusion_matrix:')
col2 = ['Actual Cat', 'Actual Dog', 'Actual Wild']
col1 = ['Predicted Cat', 'Predicted Dog', 'Predicted Wild']
mat = np.array([cat, dog, wild])
df = pd.DataFrame(mat, index=col2, columns=col1)
print(df)
print("Precision: " + str(precision_score(test_lbls, predictions, average="macro")))
print("Recall: " + str(recall_score(test_lbls, predictions, average="macro")))
print("F1: " + str(f1_score(test_lbls, predictions, average="macro")))
print('*' * 100)
model.summary()

print('*' * 100)
[test_dim_1, test_dim_2, test_dim_3, test_dim_4] = total_test.shape
samples = np.random.randint(0, test_dim_1, 15)

for k in samples:
    print('-' * 30)
    print("Sample: " + str(k))
    prediction = np.argmax(model.predict(np.reshape(total_test[k], (1, 128, 128, 3))))
    if prediction == 0:
        print("Prediction is CAT")
    elif prediction == 1:
        print("Prediction is DOG")
    elif prediction == 2:
        print("Prediction is WILD")

    ground_truth = test_lbls[k]
    if ground_truth == 0:
        print("Ground truth is CAT")
    elif ground_truth == 1:
        print("Ground truth is DOG")
    elif ground_truth == 2:
        print("Ground truth is WILD")
    plt.imshow(orig_total_test[k])
    plt.show()
