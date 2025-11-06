# 🎭 Emotion Classifier Web Application

Aplicación web con interfaz estilo ChatGPT para clasificar emociones en texto usando el modelo entrenado.

## 🚀 Inicio Rápido

### 1. Activar el entorno virtual
```bash
source venv/bin/activate
```

### 2. Instalar Flask (si no está instalado)
```bash
pip install flask
```

### 3. Ejecutar la aplicación
```bash
python app.py
```

### 4. Abrir en el navegador
```
http://localhost:5000
```

## 📋 Características

### ✨ Interfaz Usuario
- 🎨 **Diseño estilo ChatGPT** - Interfaz moderna y familiar
- 🌙 **Modo oscuro** - Colores suaves para los ojos
- 📱 **Responsive** - Funciona en móviles y tablets
- ⚡ **Tiempo real** - Resultados instantáneos

### 🧠 Capacidades del Modelo
- **6 Emociones detectables:**
  - 😊 Joy (Alegría)
  - 😢 Sadness (Tristeza)
  - ❤️ Love (Amor)
  - 😠 Anger (Enojo)
  - 😨 Fear (Miedo)
  - 😮 Surprise (Sorpresa)

### 📊 Información Mostrada
- **Emoción principal** con emoji y nombre
- **Nivel de confianza** con barra de progreso
- **Todas las probabilidades** para cada emoción
- **Colores diferenciados** por tipo de emoción

## 🎯 Cómo Usar

### Método 1: Escribir manualmente
1. Escribe tu texto en inglés en el campo de entrada
2. Presiona Enter o clic en el botón de enviar (↗️)
3. El modelo analizará el texto y mostrará la emoción detectada

### Método 2: Usar ejemplos
1. Haz clic en uno de los botones de ejemplo en la pantalla de bienvenida
2. El texto se copiará automáticamente al campo de entrada
3. Presiona Enter para analizar

## 📝 Ejemplos de Textos

### Alegría (Joy)
```
I am so happy and excited about this!
Today is the best day of my life!
I can't wait for the party tonight!
```

### Tristeza (Sadness)
```
I feel terrible and hopeless today
I miss you so much, it hurts
Everything feels empty and meaningless
```

### Amor (Love)
```
I love you more than anything
You make my heart smile
I cherish every moment with you
```

### Enojo (Anger)
```
This makes me so angry and frustrated
I can't believe you did that to me!
This is absolutely unacceptable!
```

### Miedo (Fear)
```
I'm scared and worried about what might happen
This situation terrifies me
I don't know what to do, I'm afraid
```

### Sorpresa (Surprise)
```
Wow, I didn't expect that at all!
I can't believe this is happening!
What a shocking revelation!
```

## 🔧 Arquitectura Técnica

### Backend (Flask)
- **Framework:** Flask 3.x
- **Modelo:** TF-IDF + Logistic Regression (Balanced)
- **Precisión:** 85.79% (macro)
- **F1-Score:** 88.29% (macro)

### Frontend
- **HTML5** con CSS3 integrado
- **JavaScript** vanilla (sin frameworks)
- **Diseño responsivo** con CSS Grid y Flexbox
- **Animaciones suaves** para mejor UX

### Endpoints API

#### `GET /`
- Página principal con interfaz de chat
- Template: `templates/index.html`

#### `POST /predict`
- Clasificación de texto
- **Request:**
  ```json
  {
    "text": "I am so happy today!"
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "emotion": "Joy",
    "confidence": 0.85,
    "emoji": "😊",
    "color": "#FFD700",
    "all_probabilities": [...]
  }
  ```

#### `GET /examples`
- Obtener textos de ejemplo
- **Response:**
  ```json
  {
    "examples": ["...", "..."]
  }
  ```

## 🎨 Paleta de Colores por Emoción

| Emoción | Color | Código |
|---------|-------|--------|
| Joy | Dorado | #FFD700 |
| Sadness | Azul | #4682B4 |
| Love | Rosa | #FF69B4 |
| Anger | Rojo | #FF4500 |
| Fear | Morado | #9370DB |
| Surprise | Naranja | #FF8C00 |

## 🚦 Solución de Problemas

### Error: "Address already in use"
El puerto 5000 está ocupado. Cambia el puerto en `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Error: "Model file not found"
Asegúrate de que `best_emotion_model.pkl` esté en el mismo directorio:
```bash
ls -la best_emotion_model.pkl
```
Si no existe, ejecuta `main.py` primero para entrenar el modelo.

### Error: "Flask not installed"
Instala Flask:
```bash
pip install flask
```

### La página no carga
Verifica que el servidor esté corriendo:
```bash
# Deberías ver:
# * Running on http://0.0.0.0:5000
```

## 📊 Métricas del Modelo

- **Accuracy:** 91.13%
- **Precision (Macro):** 85.79%
- **Recall (Macro):** 91.95%
- **F1-Score (Macro):** 88.29%

### Rendimiento por Emoción

| Emoción | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Sadness | 97.36% | 92.23% | 94.73% |
| Joy | 97.13% | 89.01% | 92.89% |
| Anger | 89.02% | 94.00% | 91.44% |
| Fear | 87.46% | 86.65% | 87.05% |
| Love | 74.24% | 96.34% | 83.86% |
| Surprise | 69.53% | 93.45% | 79.74% |

## 🔐 Consideraciones de Seguridad

- **Producción:** Usa un servidor WSGI como Gunicorn o uWSGI
- **Debug Mode:** Desactiva `debug=True` en producción
- **Input validation:** El modelo valida que el texto no esté vacío
- **CORS:** Agrega Flask-CORS si necesitas API pública

## 📦 Estructura de Archivos

```
.
├── app.py                      # Aplicación Flask
├── templates/
│   └── index.html             # Interfaz web
├── best_emotion_model.pkl     # Modelo entrenado
├── feature_engineering.py     # Extractores de features
├── requirements.txt           # Dependencias Python
└── README_WEBAPP.md          # Esta documentación
```

## 🚀 Despliegue en Producción

### Usando Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Usando Docker (opcional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📝 Licencia

Proyecto académico - Universidad EAFIT 2025
Inteligencia Artificial - Trabajo Final

## 👥 Autores

- Miguel Villegas
- Esteban Molina

---

**¡Disfruta clasificando emociones! 🎭**
