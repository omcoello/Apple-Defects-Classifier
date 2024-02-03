"""

Documentation for the given code:

1. Import necessary libraries/modules:
    - VGG19 from tensorflow.keras.applications: Pre-trained VGG19 model.
    - layers, models, optimizers from tensorflow.keras: Layers, models, and optimizers for building neural networks.
    - ImageDataGenerator from tensorflow.keras.preprocessing.image: Generator for generating batches of augmented data.
    - Recall from tensorflow.keras.metrics: Metric for evaluating model performance.
    - set_memory_growth, list_physical_devices from tensorflow.config.experimental: Functions for setting memory growth of GPU.
    - TensorBoard, EarlyStopping, ModelCheckpoint from tensorflow.keras.callbacks: Callbacks for training monitoring and model checkpointing.
    - time, datetime, os, cv2: Modules for time measurement, date/time operations, operating system functions, and image processing.
"""

from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Recall
from tensorflow.config.experimental import set_memory_growth, list_physical_devices
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import time, datetime, os, cv2

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
    class_mode='categorical'
)

generador_validacion = datagen.flow_from_directory(
    directorio_validacion,
    target_size=image_size,
    batch_size=8,
    class_mode='categorical'
)

generador_prueba = datagen.flow_from_directory(
    directorio_prueba,
    target_size=image_size,
    batch_size=8,
    class_mode='categorical'
)

# Crear modelo VGG-19 preentrenado
vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(960, 830, 3))

# Congelar las capas convolucionales preentrenadas
# for layer in vgg_base.layers:
#     layer.trainable = False

# Construir el modelo personalizado
model = models.Sequential()
model.add(vgg_base)
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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "vgg19_NIR"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience = 6, restore_best_weights=True)

# Crear un callback para guardar los pesos del modelo después de cada época
checkpoint_filepath = os.path.join('models/vgg19/epochs/', 'pesos_prueba_epoca_{epoch:02d}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

# Entrenar el modelo
model.fit(
    generador_entrenamiento,
    steps_per_epoch=generador_entrenamiento.samples // 8,
    epochs=25,
    validation_data=generador_prueba,
    validation_steps=generador_prueba.samples // 8,
    callbacks=[tensorboard_callback, model_checkpoint_callback]
)

results = model.evaluate(generador_prueba, steps=generador_prueba.samples // 8)

print("Precisión:", results[1])

modelPath = 'models/vgg19/vgg19_rgb.keras'
model.save(modelPath)

# Cálculo del tiempo transcurrido
execution_time = time.time() - start_time

# Imprimir el tiempo de ejecución en segundos
print(f"Tiempo de ejecución: {execution_time} segundos")
