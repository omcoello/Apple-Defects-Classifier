"""
Documentation for the given code:

1. Import necessary libraries/modules:
    - ResNet50 from tensorflow.keras.applications: Pre-trained ResNet50 model.
    - layers, models, optimizers from tensorflow.keras: Layers, models, and optimizers for building neural networks.
    - ImageDataGenerator from tensorflow.keras.preprocessing.image: Generator for generating batches of augmented data.
    - TensorBoard, EarlyStopping, ModelCheckpoint from tensorflow.keras.callbacks: Callbacks for training monitoring and model checkpointing.
    - Recall from tensorflow.keras.metrics: Metric for evaluating model performance.
    - set_memory_growth, list_physical_devices from tensorflow.config.experimental: Functions for setting memory growth of GPU.
    - time, os, cv2, datetime: Modules for time measurement, operating system functions, image processing, and date/time operations.
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Recall
from tensorflow.config.experimental import set_memory_growth, list_physical_devices
import time, os, cv2, datetime

devices = list_physical_devices('GPU')
set_memory_growth(devices[0], True)

# Inicio del temporizador
start_time = time.time()
image_size = (224,224)

# Directorios de entrenamiento y prueba
directorio_entrenamiento = 'images/train/nir'
directorio_validacion = 'images/validation/nir'
directorio_prueba = 'images/test/nir'

# Configuración del generador de datos
datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=lambda x: cv2.resize(x, (224, 224)))

# Generadores de imágenes para entrenamiento y prueba
generador_entrenamiento = datagen.flow_from_directory(
    directorio_entrenamiento,
    target_size=image_size,
    batch_size=8,
    class_mode='categorical',
    #color_mode='grayscale'
)

generador_validacion = datagen.flow_from_directory(
    directorio_validacion,
    target_size=image_size,
    batch_size=8,
    class_mode='categorical',
    #color_mode='grayscale'
)

generador_prueba = datagen.flow_from_directory(
    directorio_prueba,
    target_size=image_size,
    batch_size=8,
    class_mode='categorical',
    #color_mode='grayscale'
)

# Crear modelo ResNet-50 preentrenado
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(960, 830, 3))


# Congelar las capas convolucionales preentrenadas
# for layer in resnet_base.layers[:len(resnet_base.layers)//2]:
#     layer.trainable = False

model = models.Sequential()
model.add(resnet_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))  # 3 clases: golpes, manchas, podredumbre

custom_optimizer = optimizers.Adam(learning_rate=0.0001)

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Recall()])

# Resumen del modelo
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_resnet50_NIR_PostTrain"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience = 6, restore_best_weights=True)

# Crear un callback para guardar los pesos del modelo después de cada época
checkpoint_filepath = os.path.join('models/resnet50/post_epochs/', 'pesos_prueba_epoca_{epoch:02d}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

# Entrenar el modelo
# model.fit(
#     generador_entrenamiento,
#     steps_per_epoch=generador_entrenamiento.samples // 8,
#     epochs=25, 
#     validation_data=generador_prueba,
#     validation_steps=generador_prueba.samples // 8,
#     callbacks=[tensorboard_callback, model_checkpoint_callback]
# )

#--- codigo para cargar pesos de una determinada epoca y continuar entrenamiento desde la siguiente epoca
# Cargar los pesos después de la quinta época
# Asegúrate de proporcionar la ruta correcta del archivo de pesos

weights_file_path = 'models/resnet50/epochs/pesos_prueba_epoca_25.h5'
model.load_weights(weights_file_path)

for layer in model.layers[0].layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Recall()])

additional_epochs = 10
history_additional = model.fit(
    generador_entrenamiento,
    steps_per_epoch=generador_entrenamiento.samples // 8,
    epochs=additional_epochs,
    validation_data=generador_validacion,
    validation_steps=generador_validacion.samples // 8,
    callbacks=[tensorboard_callback, model_checkpoint_callback]
)

#----------
modelPath = 'models/resnet50/resnet50_postTrain.keras'
model.save(modelPath, save_format="tf")

results = model.evaluate(generador_prueba, steps=generador_prueba.samples // 8)
print("Resultados:",results)
print("Precisión:",results[1])
#print("MSE:",results[2])
# print("recall:",results[3])
# print("F1-Score:",results[4])


# Cálculo del tiempo transcurrido
execution_time = time.time() - start_time

# Imprimir el tiempo de ejecución en segundos
print(f"Tiempo de ejecución: {execution_time} segundos")
