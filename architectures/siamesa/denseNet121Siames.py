from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime

# Directorios de entrenamiento y prueba
directorio_entrenamiento_rgb = './images/train/rgb'
directorio_entrenamiento_sil = './images/train/nir'
directorio_prueba_rgb = './images/validation/rgb'
directorio_prueba_sil = './images/validation/nir'
directorio_test_rgb = './images/test/rgb'
directorio_test_sil = './images/test/nir'
directorio_guardado_modelos = './siames_models/rgb-nir/DenseNet121'

batch_size = 8
target_size = (960, 830)  # Ajusta el tamaño según tus requisitos

# Generador de imágenes para el entrenamiento
train_datagen = ImageDataGenerator(rescale=1./255)
train_rgb_generator = train_datagen.flow_from_directory(
    directorio_entrenamiento_rgb,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Cambia a 'categorical'
    # seed=42
)
train_sil_generator = train_datagen.flow_from_directory(
    directorio_entrenamiento_sil,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Cambia a 'categorical'
    # seed=42
)

# Generador de imágenes para la validación
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_rgb_generator = validation_datagen.flow_from_directory(
    directorio_prueba_rgb,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Cambia a 'categorical'
    # seed=42
)
validation_sil_generator = validation_datagen.flow_from_directory(
    directorio_prueba_sil,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Cambia a 'categorical'
    # seed=42
)

# Generador de imágenes para pruebas
test_datagen = ImageDataGenerator(rescale=1./255)
test_rgb_generator = test_datagen.flow_from_directory(
    directorio_test_rgb,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Cambia a 'categorical'
    # seed=42
)
test_sil_generator = test_datagen.flow_from_directory(
    directorio_test_sil,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Cambia a 'categorical'
    # seed=42
)

# Crear pares de imágenes para la red siamesa
def generate_image_pairs(generator1, generator2):
    while True:
        x1, y1 = generator1.next()
        x2, y2 = generator2.next()
        yield ([x1, x2], y1)  # Salida es la etiqueta categórica de la imagen RGB original

# Crear modelo DenseNet121 preentrenado
densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(960, 830, 3))

# Congelar las capas convolucionales preentrenadas
for layer in densenet_base.layers:
    layer.trainable = False

# Construir el modelo Siamese con DenseNet121
input_rgb = layers.Input(shape=(960, 830, 3), name='input_rgb')
input_sil = layers.Input(shape=(960, 830, 3), name='input_sil')

# Rama compartida para las imágenes RGB y siluetas
shared_branch = models.Sequential([
    densenet_base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # Agregar BatchNormalization
    layers.Dense(128, activation='relu')
], name='shared_branch')
# Salidas de cada rama
output_rgb = shared_branch(input_rgb)
output_sil = shared_branch(input_sil)

# Concatenar las salidas y agregar capas adicionales
merged = layers.concatenate([output_rgb, output_sil])

# Capas MLP para fusión y clasificación
merged = layers.Dense(64, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)
merged = layers.Dense(32, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)
output = layers.Dense(3, activation='softmax')(merged)  # 3 clases: golpes, manchas, podredumbre

# Modelo Siamese final
siamese_model = models.Model(inputs=[input_rgb, input_sil], outputs=output)

# Compilar el modelo Siamese
siamese_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo Siamese
siamese_model.summary()

# Configuración del guardado del modelo
ruta_guardado_modelo = os.path.join(directorio_guardado_modelos, 'epochs/siamese_DenseNet121_RGB_NIR_epoca_{epoch:02d}.h5')

# Callback para TensorBoard
log_dir = "logs_siames/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " DenseNet121_RGB_NIR"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Callback para detener el entrenamiento temprano
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Crea un callback para guardar el modelo después de cada época
checkpoint = ModelCheckpoint(
    ruta_guardado_modelo,
    monitor='val_accuracy',  # Puedes cambiar a la métrica que desees
    save_best_only=False,  # Guardará el modelo después de cada época
    save_weights_only=False,  # Guarda toda la arquitectura y los pesos
    mode='auto',
    verbose=1
)

# Lista de callbacks
callbacks_list = [checkpoint, tensorboard_callback]

# Entrenar el modelo Siamese con el callback
siamese_model.fit(
    generate_image_pairs(train_rgb_generator, train_sil_generator),
    steps_per_epoch=train_rgb_generator.samples // batch_size,
    epochs=25,
    validation_data=generate_image_pairs(validation_rgb_generator, validation_sil_generator),
    validation_steps=validation_rgb_generator.samples // batch_size,
    callbacks=callbacks_list
)

# Evaluación del modelo Siamese en el conjunto de prueba
siamese_results = siamese_model.evaluate(
    generate_image_pairs(test_rgb_generator, test_sil_generator),
    steps=test_rgb_generator.samples // batch_size,
    verbose=0
)

# Imprimir resultados
print("Precisión del modelo Siamese en el conjunto de prueba:", siamese_results[1])
