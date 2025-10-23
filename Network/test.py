import numpy as np

train_val_images = './DATASET/train_images.npy' # Train 80%, Validation 20%
train_val_labels = './DATASET/train_labels.npy' # Train 80%, Validation 20%


train_val_images = np.load(train_val_images)
train_val_labels = np.load(train_val_labels)

print(train_val_images)