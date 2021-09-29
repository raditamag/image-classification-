import os
from os import listdir
listdir(r"D:\projek\machine learning\image clasification\rockpaperscissors\rockpaperscissors\rps-cv-images")
base_dir = (r"D:\projek\machine learning\image clasification\rockpaperscissors\rockpaperscissors\rps-cv-images")


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(
                      rescale = 1/255,
                      rotation_range = 20,
                      horizontal_flip = True,
                      brightness_range = [1,1.5],
                      zoom_range = [1,1.5],
                      validation_split = 0.4
)
val_generator = ImageDataGenerator(
                      rescale = 1/255,
                      rotation_range = 20,
                      horizontal_flip = True,
                      brightness_range = [1,1.5],
                      zoom_range = [1,1.5],
                      validation_split = 0.4
)

train_dataset = train_generator.flow_from_directory(
                      batch_size = 8,
                      directory = base_dir,
                      shuffle = True,
                      target_size = (150,150),
                      subset = "training",
                      class_mode = 'categorical'
)

validation_dataset = val_generator.flow_from_directory(
                      batch_size = 8,
                      directory = base_dir,
                      shuffle = True,
                      target_size = (150,150),
                      subset = "validation",
                      class_mode = 'categorical'
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization

model = Sequential([
                    Conv2D(32,(3,3), activation = 'relu', input_shape = (150,150,3)),
                    

                    Conv2D(32, (3,3), activation = 'relu', padding= 'same'),
                    MaxPooling2D(),

                    Conv2D(64, (3,3), activation = 'relu'),
                    MaxPooling2D(),

                    Conv2D(64, (3,3), activation = 'relu'),
                    MaxPooling2D(),
                    
                    Conv2D(128, (3,3), activation = 'relu'),
                    MaxPooling2D(),

                    Flatten(),
                    Dropout(0.2),
                    Dense(512, activation = 'relu'),
                    
                    Dense(3, activation = 'softmax')
])



from tensorflow.keras.optimizers import SGD

model.compile(
    optimizer = SGD(learning_rate =0.01, momentum = 0.9),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

print(model.summary())

from tensorflow.keras.callbacks import Callback

class mycallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>=0.95 and logs.get('val_accuracy')>=0.95 and logs.get('val_loss')<=0.05 and logs.get('loss')<=0.05):
      print("\nAkurasi dan val_akurasi telah mencapai >90%")
      self.model.stop_training = True
callback = mycallback()

history =model.fit(
    train_dataset, 
    steps_per_epoch = 20,
    epochs = 80,
    
    validation_data = validation_dataset,
    validation_steps = 5
)


import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model accuraacy')
plt.legend()
plt.show()









