import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import display, Markdown

# Markdown display helper
def printmd(string):
    display(Markdown(string))

# Load dataset
train_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')

# Rename 'label' column to 'Label'
train_df.rename(columns={'label': 'Label'}, inplace=True)
test_df.rename(columns={'label': 'Label'}, inplace=True)

# Shuffle dataset
train_df = train_df.sample(frac=1.0).reset_index(drop=True)
test_df = test_df.sample(frac=1.0).reset_index(drop=True)

# Show dataset information
printmd(f'### Number of images in the training set: {train_df.shape[0]}')
printmd(f'### Number of images in the test set: {test_df.shape[0]}')

# Reshape and normalize images
def preprocess_images(df):
    images = df.drop('Label', axis=1).values.reshape(-1, 28, 28, 1)
    images = images / 255.0  # Normalize
    return images

X_train = preprocess_images(train_df)
X_test = preprocess_images(test_df)

y_train = train_df['Label'].values
y_test = test_df['Label'].values

# Create a validation split (10%)
val_index = int(X_train.shape[0] * 0.1)
X_val, y_val = X_train[:val_index], y_train[:val_index]
X_train, y_train = X_train[val_index:], y_train[val_index:]

# Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=10,
                                   zoom_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)
val_datagen = ImageDataGenerator()

# Train and validation generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=64)
val_generator = val_datagen.flow(X_val, y_val, batch_size=64)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')  # 25 classes (A-Z, without J or Z)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model Summary
model.summary()

# Training the model
history = model.fit(train_generator, 
                    validation_data=val_generator, 
                    epochs=30,
                    callbacks=[learning_rate_reduction, early_stopping])

# Plotting accuracy and loss
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(ax=axes[0])
axes[0].set_title('Model Accuracy')
pd.DataFrame(history.history)[['loss', 'val_loss']].plot(ax=axes[1])
axes[1].set_title('Model Loss')
plt.show()

# Evaluate on test data
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(X_test, y_test, batch_size=64, shuffle=False)
test_loss, test_accuracy = model.evaluate(test_generator)
printmd(f"## Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict labels for test data
pred = np.argmax(model.predict(X_test), axis=1)

# Classification Report
print(classification_report(y_test, pred, target_names=[chr(i) for i in range(65, 90) if chr(i) != 'J' and chr(i) != 'Z']))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[chr(i) for i in range(65, 90) if chr(i) != 'J' and chr(i) != 'Z'], 
            yticklabels=[chr(i) for i in range(65, 90) if chr(i) != 'J' and chr(i) != 'Z'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
