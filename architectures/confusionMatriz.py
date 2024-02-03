import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

directorio_prueba = 'images/test/nir'
datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=lambda x: cv2.resize(x, (224, 224)))

generador_prueba = datagen.flow_from_directory(
    directorio_prueba,
    target_size=(224,224),    
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

model = load_model('models/mobilenetv1/mobilenet.keras')

# Realiza predicciones con el modelo
y_pred = model.predict_generator(generador_prueba)

# Get real labels from generator
y_true = generador_prueba.classes
y_true_one_hot = to_categorical(y_true, num_classes=3)

# Convert predictions and real labels to classes 
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true_one_hot, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Map classes to their original name
class_names = generador_prueba.class_indices
class_names = {v: k for k, v in class_names.items()}

# Create heatmap with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[class_names[i] for i in range(3)],
            yticklabels=[class_names[i] for i in range(3)])
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas reales')
plt.title('Matriz de Confusión')
plt.show()

# Muestra el reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names.values()))
print(y_true)
print(y_pred)

print(conf_matrix.shape)
