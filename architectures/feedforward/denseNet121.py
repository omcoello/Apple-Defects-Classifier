"""
Documentation for the given code:

1. Import necessary libraries/modules:
    - tensorflow as tf: TensorFlow library.
    - ImageDataGenerator from tensorflow.keras.preprocessing.image: Generator for generating batches of augmented data.
    - DenseNet121 from tensorflow.keras.applications: Pre-trained DenseNet121 model.
    - layers, models from tensorflow.keras: Layers and models for building neural networks.
    - TensorBoard, ModelCheckpoint from tensorflow.keras.callbacks: Callbacks for training monitoring and model checkpointing.
    - Recall from tensorflow.keras.metrics: Metric for evaluating model performance.
    - time, cv2, datetime, os: Modules for time measurement, image processing, date/time operations, and operating system functions.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.metrics import Recall
import time, cv2, datetime, os

# Inicio del temporizador
start_time = time.time()
# Rutas a tus datos de entrenamiento y prueba
train_data_dir = 'images/train/nir'
valid_data_dir = 'images/validation/nir'
test_data_dir = 'images/test/nir'

# Tamaño de las imágenes de entrada
img_width, img_height = 224, 224

# Hiperparámetros
batch_size = 32
epochs = 25

# Preprocesamiento de datos
datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=lambda x: cv2.resize(x, (224, 224)))

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    #subset='training'
)

validation_generator = datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    #subset='validation'
)

test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Construcción del modelo
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
model = models.Sequential()
# Añadiendo el modelo base DenseNet121
model.add(base_model)
# Capa de pooling global para reducir la dimensionalidad
model.add(layers.GlobalAveragePooling2D())
# Capa densa con 512 unidades y activación ReLU
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.5))  # Dropout para regularización
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout para regularización
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout para regularización
model.add(layers.Dense(3, activation='softmax'))

# Congelar las capas del modelo base
# for layer in base_model.layers:
#     layer.trainable = False

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Recall()])

model.summary()

# Define log directory for TensorBoard and model checkpoint callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_densenet121_NIR_PostTrain"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#-----Model saving per epoch code
# Create callback for weight model saving per epoch
checkpoint_filepath = os.path.join('models/densenet121/post_epochs/', 'pesos_epoca_{epoch:02d}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False,  # Guardar pesos después de cada época
    verbose=1
)

# Training model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[tensorboard_callback, model_checkpoint_callback]
)

#--- codigo para cargar pesos de una determinada epoca y continuar entrenamiento desde la siguiente epoca
# Cargar los pesos después de la quinta época
# Asegúrate de proporcionar la ruta correcta del archivo de pesos

# weights_file_path = 'models/densenet121/epochs/pesos_epoca_25.h5'
# model.load_weights(weights_file_path)

# for layer in model.layers[0].layers:
#     layer.trainable = False

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Recall()])

# additional_epochs = 10
# history_additional = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=additional_epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     callbacks=[tensorboard_callback, model_checkpoint_callback]
# )

#----------

# Guardar el modelo entrenado en un directorio
modelPath = 'models/densenet121/densenet121_postTrain.keras'
model.save(modelPath, save_format="tf")

# Evaluación del modelo
results= model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Resultados: {results}')
print(f'Precisión: {results[1]}')


# #calculo de tiempo
execution_time = time.time() - start_time

# Imprimir el tiempo de ejecución en segundos
print(f"Tiempo de ejecución: {execution_time} segundos")
