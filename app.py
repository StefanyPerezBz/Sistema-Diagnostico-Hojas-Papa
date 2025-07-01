# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, Xception, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from scipy import stats
import time
import base64

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Enfermedades en Papas",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🥔 Potato Disease Classifier")
st.markdown("""
Sistema de clasificación de enfermedades en hojas de papa usando Deep Learning.
Identifica:
- **Tizón tardío** (Late blight)
- **Tizón temprano** (Early blight)
- **Marchitez bacteriana** (Bacterial wilt)
- **Hojas sanas** (Healthy)
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    dataset_path = st.text_input("Ruta del dataset", "data/PlantVillage")
    test_size = st.slider("% Validación", 10, 40, 20)
    epochs = st.slider("Épocas", 5, 100, 30)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    learning_rate = st.number_input("Learning rate", 0.00001, 0.01, 0.001, format="%.5f")

    # Modelos disponibles
    MODELS = {
        "EfficientNetB0": EfficientNetB0,
        "ResNet50V2": ResNet50V2,
        "Xception": Xception,
        "MobileNetV2": MobileNetV2
    }
    selected_models = st.multiselect("Modelos", list(MODELS.keys()), default=["EfficientNetB0"])

    # Opciones avanzadas
    with st.expander("Opciones avanzadas"):
        use_augmentation = st.checkbox("Data augmentation", True)
        use_early_stopping = st.checkbox("Early stopping", True)

# Cargar datos
@st.cache_data
def load_data(dataset_path):
    try:
        classes = sorted(os.listdir(dataset_path))
        images = []
        labels = []

        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(dataset_path, class_name)
            for img_file in os.listdir(class_path)[:500]:  # Limitar para demo
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(class_idx)

        X = np.array(images, dtype='float32')
        y = np.array(labels)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y)

        # Normalizar
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # One-hot encoding
        y_train = to_categorical(y_train, len(classes))
        y_test = to_categorical(y_test, len(classes))

        return X_train, X_test, y_train, y_test, classes

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

# Función para crear modelo
def create_model(base_model_name, num_classes, learning_rate):
    base_model = MODELS[base_model_name](
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3))
    
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')])
    return model

# Entrenar modelos
def train_models(X_train, X_test, y_train, y_test, classes):
    results = {}
    models = {}
    histories = {}
    training_times = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_name in enumerate(selected_models):
        start_time = time.time()
        status_text.text(f"Entrenando {model_name}...")

        try:
            model = create_model(model_name, len(classes), learning_rate)

            callbacks = []
            if use_early_stopping:
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True))

            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001))

            # Data augmentation
            train_datagen = ImageDataGenerator(
                rotation_range=25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest') if use_augmentation else None

            # Entrenamiento
            if train_datagen:
                history = model.fit(
                    train_datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0)
            else:
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0)

            # Evaluación
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)

            # Métricas
            results[model_name] = {
                'accuracy': accuracy_score(y_test_classes, y_pred_classes),
                'precision': precision_score(y_test_classes, y_pred_classes, average='weighted'),
                'recall': recall_score(y_test_classes, y_pred_classes, average='weighted'),
                'f1': f1_score(y_test_classes, y_pred_classes, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred, multi_class='ovr'),
                'model': model,
                'history': history.history
            }

            models[model_name] = model
            histories[model_name] = history.history
            training_times[model_name] = time.time() - start_time

            # Guardar modelo
            os.makedirs('models', exist_ok=True)
            model_path = f"models/potato_{model_name}.h5"
            model.save(model_path)

            progress_bar.progress((i + 1) / len(selected_models))

        except Exception as e:
            st.error(f"Error entrenando {model_name}: {str(e)}")
            continue

    status_text.text("✅ Entrenamiento completado!")
    time.sleep(1)
    status_text.empty()

    return results, models, histories, training_times

# Generar reporte PDF
def generate_report(results, classes, filename="reports/report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título
    title = Paragraph("Reporte de Clasificación de Enfermedades en Papas", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Tabla de resultados
    data = [["Modelo", "Accuracy", "F1-Score", "ROC AUC"]]
    for model_name, metrics in results.items():
        data.append([
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['roc_auc']:.4f}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 24))

    doc.build(story)
    return filename

# Cargar datos
X_train, X_test, y_train, y_test, classes = load_data(dataset_path)

# Estado de la sesión
if 'models' not in st.session_state:
    st.session_state.models = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Interfaz principal
if X_train is not None:
    st.subheader("📊 Estadísticas del Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Muestras entrenamiento:** {len(X_train)}")
        st.write(f"**Muestras validación:** {len(X_test)}")
        st.write(f"**Clases:** {', '.join(classes)}")
    
    with col2:
        fig, ax = plt.subplots()
        class_dist = np.sum(y_train, axis=0)
        ax.bar(classes, class_dist)
        ax.set_title("Distribución de Clases")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Botón de entrenamiento
    if st.button("🚀 Entrenar Modelos"):
        with st.spinner("Entrenando..."):
            results, models, histories, training_times = train_models(
                X_train, X_test, y_train, y_test, classes)
        
        if results:
            st.session_state.models = models
            st.session_state.results = results
            
            st.subheader("📈 Resultados")
            df_results = pd.DataFrame.from_dict({
                model: {
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1']
                }
                for model, metrics in results.items()
            }, orient='index')
            
            st.dataframe(df_results.style.format("{:.2%}"))
            
            # Gráfico de comparación
            fig, ax = plt.subplots(figsize=(10, 5))
            df_results.plot(kind='bar', ax=ax)
            ax.set_title("Comparación de Modelos")
            ax.set_ylabel("Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Generar reporte
            report_path = generate_report(results, classes)
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Descargar Reporte",
                    f,
                    file_name="reporte_enfermedades_papa.pdf",
                    mime="application/pdf"
                )

    # Sección de predicción
    st.subheader("🔍 Clasificar Nueva Imagen")
    uploaded_file = st.file_uploader("Sube una imagen de hoja de papa", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and st.session_state.models:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", width=300)
        
        # Preprocesar
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicciones
        predictions = {}
        for model_name, model in st.session_state.models.items():
            pred = model.predict(img_array)
            predictions[model_name] = pred[0]
        
        # Mostrar resultados
        st.subheader("📋 Diagnóstico")
        cols = st.columns(len(predictions))
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            with cols[idx]:
                st.write(f"**{model_name}**")
                prob_df = pd.DataFrame({
                    'Clase': classes,
                    'Probabilidad': pred
                }).sort_values('Probabilidad', ascending=False)
                
                st.dataframe(prob_df.style.format({'Probabilidad': '{:.2%}'}))
                
                top_class = prob_df.iloc[0]
                st.metric(
                    label="Predicción",
                    value=top_class['Clase'],
                    delta=f"{top_class['Probabilidad']:.2%} confianza"
                )

# Footer
st.markdown("---")
st.markdown("""
**Potato Disease Classifier** 🥔  
*Sistema de clasificación de enfermedades en hojas de papa usando Deep Learning*  
[GitHub Repo](https://github.com/StefanyPerezBz/potato-disease-detection)
""")