import tensorflow as tf
print(tf.__version__)
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline


!wget --no-check-certificate \
  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip


local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/rockpaperscissors'
train_dir = os.path.join(base_dir, 'rps-cv-images')

os.listdir('/tmp/rockpaperscissors')
os.listdir('/tmp/rockpaperscissors/rps-cv-images')



# Augmented Training
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    shear_range=0.2,
                    zoom_range=0.2,
                    fill_mode='nearest',
                    validation_split = 0.4
                     )

train_generator = train_datagen.flow_from_directory(
        train_dir,                 
        target_size=(100, 150),    
        batch_size=32,
        class_mode='categorical',
        subset='training')

train_generator.class_indices

# Augmented Validation
validation_datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split = 0.4)

validation_generator = validation_datagen.flow_from_directory(
        train_dir,                 
        target_size=(100, 150),   
        batch_size=32, 
        class_mode='categorical',
        subset='validation')

validation_generator.class_indices


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])


class myCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('accuracy') >= 0.95:
      print('\nEpoch', epoch, '\nAccuracy has reached = %2.2f%%' %(logs['accuracy']*100), 'stop the training process')
      self.model.stop_training = True

history = model.fit(
      train_generator,
      steps_per_epoch=25,                   
      epochs=20,
      validation_data=validation_generator, 
      validation_steps=5,                   
      verbose=2,
      callbacks = [myCallbacks()]
      )



plt.figure(figsize=(14, 5))
# Accuracy Plot
plt.subplot(1, 2, 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()



uploaded = files.upload()
# Predict Images
for fn in uploaded.keys():
    path = fn
    img_predict = image.load_img(path, target_size = (100, 150))
    imgplot = plt.imshow(img_predict)
    x = image.img_to_array(img_predict)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size = 10)

    print(fn)
    if classes[0][0] == 1.0:
        print("This image forms Paper")
    elif classes[0][1] == 1.0:
        print("This image forms Rock")
    else:
        print("This image forms Scissors")
classes