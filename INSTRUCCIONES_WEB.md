# 🎭 Aplicación Web de Clasificación de Emociones

## ✅ Estado: COMPLETADO Y FUNCIONANDO

La aplicación web con interfaz estilo ChatGPT está **lista y operativa**.

## 🚀 Cómo Iniciar la Aplicación

### Pasos para ejecutar:

1. **Abrir terminal** en el directorio del proyecto

2. **Activar el entorno virtual:**
   ```bash
   source venv/bin/activate
   ```

3. **Ejecutar la aplicación:**
   ```bash
   python app.py
   ```

4. **Abrir navegador** y visitar:
   ```
   http://localhost:5000
   ```

## 🎨 Características de la Interfaz

### Diseño Estilo ChatGPT
- ✅ Fondo oscuro (#343541)
- ✅ Área de chat con mensajes diferenciados
- ✅ Usuario (morado) vs Asistente (verde)
- ✅ Input en la parte inferior con botón de envío
- ✅ Animaciones suaves
- ✅ 100% Responsive (móvil y desktop)

### Funcionalidades
- ✅ Entrada de texto en tiempo real
- ✅ Predicción instantánea al enviar
- ✅ Muestra emoción principal con emoji
- ✅ Nivel de confianza con barra visual
- ✅ Probabilidades de todas las emociones
- ✅ Colores diferenciados por emoción
- ✅ Botones de ejemplo para pruebas rápidas

## 📊 Prueba Realizada

**Texto probado:** "I am so happy and excited today!"

**Resultado:**
- ✅ Emoción detectada: **Joy** 😊
- ✅ Confianza: **98.18%**
- ✅ Otras probabilidades:
  - Love: 0.69%
  - Anger: 0.40%
  - Fear: 0.32%
  - Sadness: 0.23%
  - Surprise: 0.19%

## 🎯 Cómo Usar la Aplicación

### Método 1: Escribir texto personalizado
1. Escribe tu texto en inglés en el campo inferior
2. Presiona **Enter** o clic en el botón **↗️**
3. El modelo analizará y mostrará la emoción

### Método 2: Usar ejemplos predefinidos
1. Al cargar la página verás 4 botones de ejemplo
2. Haz clic en cualquiera para cargar el texto
3. Presiona Enter para analizar

## 📁 Archivos Creados

```
📦 Proyecto
├── app.py                    # ✅ Aplicación Flask
├── templates/
│   └── index.html           # ✅ Interfaz estilo ChatGPT
├── README_WEBAPP.md         # ✅ Documentación completa
├── INSTRUCCIONES_WEB.md    # ✅ Este archivo
└── test_request.json       # ✅ Archivo de prueba
```

## 🌐 URLs Disponibles

### Página Principal
```
http://localhost:5000/
```
Interfaz de chat completa

### API Endpoint
```
POST http://localhost:5000/predict
Content-Type: application/json
Body: {"text": "Your text here"}
```

### Ejemplos
```
GET http://localhost:5000/examples
```
Retorna textos de ejemplo

## 🧪 Pruebas Adicionales

### Probar desde terminal:
```bash
# Con curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json

# O crear tu propio JSON
echo '{"text":"I love this project!"}' > my_test.json
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @my_test.json
```

### Probar desde Python:
```python
import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={'text': 'I am so happy!'}
)
print(response.json())
```

## 📱 Screenshots del Diseño

La interfaz incluye:

### Header (Superior)
```
🎭 Emotion Classifier AI   [TF-IDF + LogReg]   F1: 88.29%
```

### Área de Bienvenida
```
Emotion Classifier AI

Escribe cualquier texto en inglés y analizaré
la emoción que expresa. Puedo detectar:
Alegría, Tristeza, Amor, Enojo, Miedo y Sorpresa.

[Botones de ejemplo]
```

### Mensajes de Chat
```
👤  I am so happy and excited today!

🤖  😊 Joy
    Confidence: 98.18%
    [Barra de progreso]

    All Emotions:
    😊 Joy      ████████ 98.18%
    ❤️  Love    ▏        0.69%
    😠 Anger   ▏        0.40%
    ...
```

### Input (Inferior)
```
┌────────────────────────────────────┐
│ Write a message in English...     │  [↗️]
└────────────────────────────────────┘
```

## 🎨 Paleta de Colores

| Elemento | Color | Uso |
|----------|-------|-----|
| Fondo principal | #343541 | Background |
| Fondo header | #202123 | Header |
| Fondo mensajes usuario | #343541 | User msg |
| Fondo mensajes asistente | #444654 | Bot msg |
| Input | #40414f | Input field |
| Acento principal | #10a37f | Buttons |
| Joy | #FFD700 | Dorado |
| Sadness | #4682B4 | Azul |
| Love | #FF69B4 | Rosa |
| Anger | #FF4500 | Rojo |
| Fear | #9370DB | Morado |
| Surprise | #FF8C00 | Naranja |

## ⚡ Rendimiento

- **Carga inicial:** < 1 segundo
- **Tiempo de predicción:** < 100ms
- **Modelo F1-Score:** 88.29%
- **Precisión:** 85.79%

## 🛑 Cómo Detener el Servidor

Para detener el servidor Flask:
1. En la terminal donde corre `app.py`
2. Presiona **Ctrl + C**

## 📝 Notas Importantes

### ✅ Lo que FUNCIONA:
- Interfaz completa estilo ChatGPT
- Predicción de emociones en tiempo real
- Visualización de probabilidades
- Responsive design
- Animaciones suaves
- Ejemplos interactivos

### 📌 Para Producción:
Si quieres desplegar en producción:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 🔒 Seguridad:
- El `debug=True` es solo para desarrollo
- Para producción, cambia a `debug=False`
- Considera agregar autenticación si es necesario

## 🎉 ¡Listo para Usar!

La aplicación está **100% funcional** y lista para:
- ✅ Demostraciones
- ✅ Presentaciones
- ✅ Pruebas de usuario
- ✅ Evaluación académica

**¡Disfruta tu clasificador de emociones estilo ChatGPT! 🎭**

---

**Proyecto:** Inteligencia Artificial - Trabajo Final
**Universidad:** EAFIT 2025
**Autores:** Miguel Villegas y Esteban Molina
