"""
Documentation for the given code:

1. Import necessary libraries/modules:
    - MobileNet from tensorflow.keras.applications: Pre-trained MobileNet model.
    - layers, models from tensorflow.keras: Layers and models for building neural networks.
    - ImageDataGenerator from tensorflow.keras.preprocessing.image: Generator for generating batches of augmented data.
    - TensorBoard, EarlyStopping, ModelCheckpoint from tensorflow.keras.callbacks: Callbacks for training monitoring and model checkpointing.
    - Recall from tensorflow.keras.metrics: Metric for evaluating model performance.
    - os, cv2, datetime, time: Modules for operating system functions, image processing, date/time operations, and time measurement.
"""

from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Recall
import os, cv2, datetime, time

start_time = time.time()

# Training, validation and testing directories
directorio_entrenamiento = 'images/train/nir'
directorio_validacion = 'images/validation/nir'
directorio_prueba = 'images/test/nir'

# Data generator setting
datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=lambda x: cv2.resize(x, (224, 224)))

# Training, validation and testing set generators
batch_size = 32
image_size = (224, 224)
generador_entrenamiento = datagen.flow_from_directory(
    directorio_entrenamiento,
    target_size= image_size,
    batch_size=batch_size,
    class_mode='categorical',
    #color_mode='grayscale'
)

generador_validacion = datagen.flow_from_directory(
    directorio_validacion,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    #color_mode='grayscale'
)

generador_prueba = datagen.flow_from_directory(
    directorio_prueba,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    #color_mode='grayscale'
)

# Create pretrained MobileNetV1 model 
mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Build personal model
model = models.Sequential()
model.add(mobilenet_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))  # 3 clases: golpes, manchas, podredumbre

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Recall()])

# Model summary
model.summary()

# Define log directory for TensorBoard, early stopping criteria, and model checkpoint callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " mobileNet_Resized_NIR"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Create callback for weight model saving per epoch
checkpoint_filepath = os.path.join('models/resized/mobilenetv1/epochs/', 'pesos_prueba_epoca_{epoch:02d}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

# Train model
model.fit(
    generador_entrenamiento,
    steps_per_epoch=generador_entrenamiento.samples // batch_size,
    epochs=25,
    validation_data=generador_validacion,
    validation_steps=generador_validacion.samples // batch_size,
    callbacks=[tensorboard_callback, model_checkpoint_callback]
)

# Evaluate model with test set
results = model.evaluate(generador_prueba, steps=generador_prueba.samples // batch_size)
print("Precisión en el conjunto de prueba:", results[1])

# Save model
modelPath = 'models/resized/mobilenetv1/mobilenet_resized.keras'
model.save(modelPath, save_format="tf")

# Calculation of elapsed time
execution_time = time.time() - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")
