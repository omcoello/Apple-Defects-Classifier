from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model('models/resized/mobilenetv1/mobilenet_resized.keras')

# load and preprocess image
ruta_imagen = "image.png"  
imagen_prueba = image.load_img(ruta_imagen, target_size=(224, 224)) 
imagen_prueba = image.img_to_array(imagen_prueba)

test_image = cv2.imread(ruta_imagen)
test_image = cv2.resize(test_image,(224,224))

test_image = np.expand_dims(imagen_prueba, axis=0)

test_image = test_image/255

# Predict image
prediccion = model.predict(test_image)

print(prediccion)

# Decode prediction to classes
etiqueta_predicha = np.argmax(prediccion)
print("Clase predicha:", etiqueta_predicha)