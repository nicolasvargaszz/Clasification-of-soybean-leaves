import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
                                     BatchNormalization, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as A  # Importación de Albumentations

# Establecer semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Rutas y parámetros
train_images_path = 'train_images'       # Carpeta con imágenes de entrenamiento
train_masks_path = 'train_masks'         # Carpeta con máscaras de entrenamiento
test_images_path = 'test_images'         # Carpeta con imágenes de prueba
output_images_path = 'output_images'     # Carpeta para guardar imágenes de salida
model_filename = 'segmentation_model.h5' # Nombre del archivo del modelo entrenado

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
NUM_CLASSES = 5  # 4 categorías + fondo

# Asegurarse de que la carpeta de salida exista
os.makedirs(output_images_path, exist_ok=True)

# Categorías y etiquetas
categories = ['background', 'good', 'bad', 'slightly_good', 'slightly_bad']
category_to_label = {category: idx for idx, category in enumerate(categories)}
label_to_category = {idx: category for category, idx in category_to_label.items()}

# Función para extraer rangos de color de las máscaras usando percentiles
def extract_color_ranges(mask_folder):
    color_ranges = {}
    for category in categories[1:]:  # Excluir 'background'
        category_path = os.path.join(mask_folder, category)
        if not os.path.isdir(category_path):
            print(f"No se encontró la carpeta para la categoría '{category}'.")
            continue
        hsv_values = []
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            # Convertir a HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_values.append(hsv_image.reshape(-1, 3))
        if hsv_values:
            hsv_values = np.vstack(hsv_values)
            # Calcular percentiles para evitar valores atípicos
            h_lower = np.percentile(hsv_values[:, 0], 5)
            h_upper = np.percentile(hsv_values[:, 0], 95)
            s_lower = np.percentile(hsv_values[:, 1], 5)
            s_upper = np.percentile(hsv_values[:, 1], 95)
            v_lower = np.percentile(hsv_values[:, 2], 5)
            v_upper = np.percentile(hsv_values[:, 2], 95)

            lower_bound = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
            upper_bound = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)

            color_ranges[category] = (lower_bound, upper_bound)
            print(f"Categoría '{category}':")
            print(f"  HSV Inferior: {lower_bound}")
            print(f"  HSV Superior: {upper_bound}")
        else:
            print(f"No se encontraron imágenes válidas para la categoría '{category}'.")
    return color_ranges

# Función para generar máscaras para las imágenes de entrenamiento
def generate_masks_for_images(image_folder, color_ranges):
    images = []
    masks = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(valid_extensions):
            continue
        filepath = os.path.join(image_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue
        # Preprocesamiento: aplicar filtro bilateral para reducir el ruido
        image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        # Redimensionar imagen
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        images.append(image)
        # Crear máscara
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        for category, (lower_bound, upper_bound) in color_ranges.items():
            category_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask[category_mask > 0] = category_to_label[category]
        masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

# Función para construir un modelo U-Net mejorado
def build_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = tf.keras.activations.relu(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = tf.keras.activations.relu(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = tf.keras.activations.relu(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = tf.keras.activations.relu(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = tf.keras.activations.relu(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = tf.keras.activations.relu(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = tf.keras.activations.relu(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = tf.keras.activations.relu(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = tf.keras.activations.relu(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = tf.keras.activations.relu(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(512, (2, 2), padding='same')(u6)
    u6 = BatchNormalization()(u6)
    u6 = tf.keras.activations.relu(u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = tf.keras.activations.relu(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = tf.keras.activations.relu(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(256, (2, 2), padding='same')(u7)
    u7 = BatchNormalization()(u7)
    u7 = tf.keras.activations.relu(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = tf.keras.activations.relu(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = tf.keras.activations.relu(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Conv2D(128, (2, 2), padding='same')(u8)
    u8 = BatchNormalization()(u8)
    u8 = tf.keras.activations.relu(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = tf.keras.activations.relu(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = tf.keras.activations.relu(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Conv2D(64, (2, 2), padding='same')(u9)
    u9 = BatchNormalization()(u9)
    u9 = tf.keras.activations.relu(u9)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = tf.keras.activations.relu(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = tf.keras.activations.relu(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Función alternativa para segmentación basada en color sin red neuronal
def segment_image_by_color(image, color_ranges):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for category, (lower_bound, upper_bound) in color_ranges.items():
        category_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        mask = cv2.bitwise_or(mask, category_mask)
    return mask

# Función para calcular IoU
def iou_metric(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.logical_and(y_true > 0, y_pred > 0)
    union = np.logical_or(y_true > 0, y_pred > 0)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Función para crear un generador de datos con Albumentations
def create_train_generator(x_set, y_set, batch_size):
    # Definir la pipeline de augmentación
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(p=0.2),
        # Puedes agregar más transformaciones si lo deseas
    ], additional_targets={'mask': 'mask'})

    dataset_size = len(x_set)
    while True:
        indices = np.random.permutation(dataset_size)
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = []
            batch_y = []
            for idx in batch_indices:
                image = x_set[idx]
                mask = y_set[idx]
                # Aplicar augmentación
                augmented = transform(image=image, mask=mask)
                batch_x.append(augmented['image'])
                batch_y.append(augmented['mask'])
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y

# Script principal
if __name__ == '__main__':
    # Paso 1: Extraer rangos de color de las máscaras
    print("Extrayendo rangos de color de las máscaras...")
    color_ranges = extract_color_ranges(train_masks_path)

    # Paso 2: Verificar si el modelo ya está entrenado y guardado
    if os.path.exists(model_filename):
        print(f"Cargando el modelo entrenado desde {model_filename}...")
        model = load_model(model_filename)
    else:
        print("Entrenando un nuevo modelo...")
        # Generar máscaras para las imágenes de entrenamiento
        images, masks = generate_masks_for_images(train_images_path, color_ranges)

        # Visualizar algunas máscaras de entrenamiento
        print("Visualizando algunas máscaras de entrenamiento...")
        for idx in range(min(3, len(images))):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
            plt.title('Imagen de Entrenamiento')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(masks[idx], cmap='jet')
            plt.title('Máscara de Entrenamiento')
            plt.axis('off')
            plt.show()

            unique_labels = np.unique(masks[idx])
            print(f"Etiquetas únicas en la máscara {idx}: {unique_labels}")

        # Normalizar imágenes y convertir máscaras a enteros
        images = images / 255.0
        masks_int = masks.astype(np.uint8)

        # División de datos
        x_train, x_val, y_train, y_val = train_test_split(images, masks_int, test_size=0.2, random_state=42)

        # Paso 3: Construir, compilar y entrenar el modelo
        model = build_unet_model()
        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Tamaño de lote
        batch_size = 8

        # Crear generador de entrenamiento
        train_generator = create_train_generator(x_train, y_train, batch_size)

        # Calcular steps_per_epoch
        steps_per_epoch = len(x_train) // batch_size

        # Entrenamiento
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=(x_val, y_val),
            epochs=50,
            callbacks=[reduce_lr, early_stopping]
        )

        # Visualizar el historial de entrenamiento
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de Validación')
        plt.legend()
        plt.title('Pérdida')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
        plt.legend()
        plt.title('Precisión')

        plt.show()

        # Guardar el modelo
        model.save(model_filename)
        print(f"Modelo guardado como {model_filename}")

    # Paso 4: Procesar imágenes de prueba
    print("Procesando imágenes de prueba...")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    test_image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(valid_extensions)]

    for test_image in test_image_files:
        test_img_path = os.path.join(test_images_path, test_image)
        test_img = cv2.imread(test_img_path)
        if test_img is None:
            print(f"No se pudo cargar la imagen de prueba: {test_img_path}")
            continue
        original_size = (test_img.shape[1], test_img.shape[0])  # (ancho, alto)
        test_img_resized = cv2.resize(test_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        test_img_input = np.expand_dims(test_img_resized, axis=0)

        # Predecir máscara
        pred_mask = model.predict(test_img_input)[0]
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)

        # Imprimir etiquetas únicas en la máscara predicha
        unique_labels = np.unique(pred_mask_resized)
        print(f"Etiquetas únicas en la máscara predicha: {unique_labels}")

        # Clasificación basada en la categoría predominante
        category_counts = {}
        for category in categories[1:]:  # Excluir 'background'
            label = category_to_label[category]
            count = np.sum(pred_mask_resized == label)
            category_counts[category] = count

        # Determinar la categoría con mayor cantidad de píxeles
        classification = max(category_counts, key=category_counts.get)
        print(f"La imagen '{test_image}' ha sido clasificada como: {classification}")

        # Visualizar la máscara predicha
        plt.figure(figsize=(10, 10))
        plt.imshow(pred_mask_resized, cmap='jet')
        plt.title(f"Máscara Predicha - {test_image}")
        plt.axis('off')
        plt.show()

        # Crear imagen de salida
        output_img = test_img.copy()  # Copia de la imagen original
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # Imagen en blanco y negro
        gray_img_colored = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Crear máscara binaria para los píxeles reconocidos
        recognized_mask = (pred_mask_resized > 0).astype(np.uint8)

        # Aplicar operaciones morfológicas para mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
        recognized_mask = cv2.morphologyEx(recognized_mask, cv2.MORPH_OPEN, kernel)
        recognized_mask = cv2.morphologyEx(recognized_mask, cv2.MORPH_CLOSE, kernel)

        # Mantener colores originales en píxeles reconocidos y convertir a gris los demás
        output_img = np.where(recognized_mask[..., np.newaxis], test_img, gray_img_colored)

        # Guardar la imagen de salida
        output_image_path = os.path.join(output_images_path, f"output_{test_image}")
        cv2.imwrite(output_image_path, output_img)
        print(f"Imagen de salida guardada en {output_image_path}")

        # Visualizar la imagen de salida
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Salida Segmentada - {test_image}")
        plt.axis('off')
        plt.show()

    # Paso 5: Implementar segmentación basada en colores (opcional)
    print("Aplicando segmentación basada en colores directamente (sin red neuronal)...")
    for test_image in test_image_files:
        test_img_path = os.path.join(test_images_path, test_image)
        test_img = cv2.imread(test_img_path)
        if test_img is None:
            print(f"No se pudo cargar la imagen de prueba: {test_img_path}")
            continue

        # Preprocesamiento: aplicar filtro bilateral
        test_img_filtered = cv2.bilateralFilter(test_img, d=9, sigmaColor=75, sigmaSpace=75)

        # Segmentar la imagen usando los rangos de color
        mask = segment_image_by_color(test_img_filtered, color_ranges)

        # Aplicar operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Crear imagen de salida
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        gray_img_colored = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        output_img_color = np.where(mask[..., np.newaxis], test_img, gray_img_colored)

        # Guardar y mostrar la imagen de salida
        output_image_path = os.path.join(output_images_path, f"output_color_{test_image}")
        cv2.imwrite(output_image_path, output_img_color)
        print(f"Imagen de salida (segmentación por color) guardada en {output_image_path}")

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(output_img_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Segmentación por Color - {test_image}")
        plt.axis('off')
        plt.show()
