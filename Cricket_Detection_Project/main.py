# Import necessary libraries for numerical operations, image processing, machine learning, and file system manipulation
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# Define paths to the dataset and where the model and binarizer will be saved after training
Datapath = r"C:\Python\Cricket_Detection_Project\dataSet"
OutputModel = r"C:\Python\Cricket_Detection_Project\TrainModel\VideoClassificationModel.h5"
OutputBinarizer = r"C:\Python\Cricket_Detection_Project\TrainModel\VideoClassificationBinarizer.pickle"

# Set of classes for classification - images not in these categories will be ignored
Sports_labels = {'cricket', 'swimming', 'boxing'}

# Inform the user that the image loading process has begun
print("Images are being loaded...")

# Get the list of all image paths in the dataset directory
pathtoImages = list(paths.list_images(Datapath))

# Initialize lists to hold image data and corresponding labels
data = []
labels = []

# Loop over all image paths
for imagePath in pathtoImages:
    # Extract the label from the path and check if it is one of the predefined labels
    label = imagePath.split(os.path.sep)[-2]
    if label not in Sports_labels:
        continue

    # Read the image from disk, convert it to RGB color space, and resize it to 224x224
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Changed from 244 to 224 to match ResNet50 input

    # Append the processed image and its label to the respective lists
    data.append(image)
    labels.append(label)

# Convert the data and labels lists to NumPy arrays for further processing
data = np.array(data)
labels = np.array(labels)

# Initialize the LabelBinarizer and transform the labels into one-hot encoding format
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split the dataset into training and testing sets
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# Define image data augmentation configurations
trainingAugmentation = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
validationAugmentation = ImageDataGenerator()

# Define the ResNet50 base model
baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Create the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# Place the headModel on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them so they will not be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
opt = SGD(learning_rate=0.0001, momentum=0.9, decay=1e-4 / 25)  # Assuming 25 epochs, adjust as needed
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
History = model.fit(
    trainingAugmentation.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=validationAugmentation.flow(X_test, y_test),
    validation_steps=len(X_test) // 32,
    epochs=25  # Set the number of epochs
)

# Save the model and LabelBinarizer
model.save(OutputModel)

# Save the LabelBinarizer to a pickle file
with open(OutputBinarizer, "wb") as file:
    pickle.dump(lb, file)

print("Model and LabelBinarizer have been saved.")
