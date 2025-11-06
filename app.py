"""
Aplicación Web Flask para Clasificación de Emociones
Interfaz estilo ChatGPT
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Cargar el modelo al iniciar la aplicación
print("Cargando modelo de clasificación de emociones...")
model_data = joblib.load('best_emotion_model.pkl')
print(f"✅ Modelo cargado: {model_data['model_name']}")

# Extraer componentes del modelo
vectorizer = model_data.get('vectorizer') or model_data.get('feature_union')
classifier = model_data['classifier']
emotion_names = model_data['emotion_names']
metrics = model_data['metrics']

# Mapeo de emociones a emojis y colores
emotion_info = {
    'Joy': {'emoji': '😊', 'color': '#FFD700'},
    'Sadness': {'emoji': '😢', 'color': '#4682B4'},
    'Love': {'emoji': '❤️', 'color': '#FF69B4'},
    'Anger': {'emoji': '😠', 'color': '#FF4500'},
    'Fear': {'emoji': '😨', 'color': '#9370DB'},
    'Surprise': {'emoji': '😮', 'color': '#FF8C00'}
}

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html',
                         model_name=model_data['model_name'],
                         f1_score=metrics['f1_macro'],
                         precision=metrics['precision_macro'])

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para hacer predicciones"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'error': 'Por favor ingresa un texto',
                'success': False
            }), 400

        # Preprocesar y predecir
        X = vectorizer.transform([text])
        prediction = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]

        # Obtener nombre de la emoción
        emotion = emotion_names[prediction]
        confidence = float(probabilities[prediction])

        # Extraer palabras más importantes (Feature Importance)
        important_words = get_important_words(text, X, prediction)

        # Preparar todas las probabilidades
        all_probs = []
        for idx, prob in enumerate(probabilities):
            em_name = emotion_names[idx]
            all_probs.append({
                'emotion': em_name,
                'probability': float(prob),
                'emoji': emotion_info.get(em_name, {}).get('emoji', ''),
                'color': emotion_info.get(em_name, {}).get('color', '#666')
            })

        # Ordenar por probabilidad
        all_probs.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'emoji': emotion_info.get(emotion, {}).get('emoji', ''),
            'color': emotion_info.get(emotion, {}).get('color', '#666'),
            'all_probabilities': all_probs,
            'important_words': important_words,
            'timestamp': datetime.now().strftime('%H:%M')
        })

    except Exception as e:
        return jsonify({
            'error': f'Error al procesar: {str(e)}',
            'success': False
        }), 500

def get_important_words(text, X_transformed, predicted_class):
    """
    Extrae las palabras más importantes para la predicción.
    Usa los coeficientes del modelo y los valores TF-IDF.
    """
    try:
        # Obtener nombres de features del vectorizador
        feature_names = vectorizer.get_feature_names_out()

        # Obtener coeficientes del modelo para la clase predicha
        if hasattr(classifier, 'coef_'):
            coefficients = classifier.coef_[predicted_class]
        else:
            # Si no tiene coef_ (ej: algunos modelos), retornar vacío
            return []

        # Obtener valores TF-IDF del texto
        tfidf_scores = X_transformed.toarray()[0]

        # Calcular importancia: coef * tfidf_score
        word_importance = []
        for idx, (coef, tfidf) in enumerate(zip(coefficients, tfidf_scores)):
            if tfidf > 0:  # Solo palabras presentes en el texto
                importance = abs(coef * tfidf)
                word_importance.append({
                    'word': feature_names[idx],
                    'importance': float(importance),
                    'coefficient': float(coef)
                })

        # Ordenar por importancia y tomar top 5
        word_importance.sort(key=lambda x: x['importance'], reverse=True)
        top_words = word_importance[:5]

        # Normalizar importancias para visualización (0-100)
        if top_words:
            max_importance = top_words[0]['importance']
            if max_importance > 0:
                for word in top_words:
                    word['importance_normalized'] = (word['importance'] / max_importance) * 100

        return top_words

    except Exception as e:
        print(f"Error extracting important words: {e}")
        return []

@app.route('/examples', methods=['GET'])
def get_examples():
    """Obtener textos de ejemplo"""
    examples = [
        "I am so happy and excited about this!",
        "I feel terrible and hopeless today",
        "I love spending time with you",
        "This makes me so angry and frustrated",
        "I'm scared and worried about what might happen",
        "Wow, I didn't expect that at all!"
    ]
    return jsonify({'examples': examples})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Iniciando servidor Flask - Clasificador de Emociones")
    print("="*80)
    print(f"📊 Modelo: {model_data['model_name']}")
    print(f"📈 F1-Score: {metrics['f1_macro']:.2%}")

    # Configuración para desarrollo y producción
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'

    print(f"🌐 Accede a: http://localhost:{port}")
    print("="*80 + "\n")

    app.run(debug=debug_mode, host='0.0.0.0', port=port)
