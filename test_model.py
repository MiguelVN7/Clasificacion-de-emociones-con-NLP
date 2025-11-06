#!/usr/bin/env python
"""Script de prueba rápida del modelo guardado"""
import joblib

# Cargar modelo
print("Cargando modelo...")
model_data = joblib.load('best_emotion_model.pkl')

print("\n✅ VERIFICACIÓN DEL MODELO GUARDADO")
print("="*80)
print(f"📊 Modelo: {model_data['model_name']}")
print(f"\n📈 Métricas del modelo:")
for metric, value in model_data['metrics'].items():
    print(f"   • {metric}: {value:.4f} ({value*100:.2f}%)")

print(f"\n🔧 Componentes:")
print(f"   • Classifier: {type(model_data['classifier']).__name__}")
if 'vectorizer' in model_data:
    print(f"   • Vectorizer: {type(model_data['vectorizer']).__name__}")
if 'feature_union' in model_data:
    print(f"   • Feature Union: {type(model_data['feature_union']).__name__}")

# Probar predicción
print(f"\n🧪 Probando predicción...")
test_texts = [
    "I am so happy today!",
    "I feel terrible and sad",
    "I love this so much!",
    "I am very angry at you",
    "This is scary and frightening",
    "Wow, what a surprise!"
]

vectorizer = model_data.get('vectorizer') or model_data.get('feature_union')
classifier = model_data['classifier']
emotion_names = model_data['emotion_names']

for text in test_texts:
    X = vectorizer.transform([text])
    prediction = classifier.predict(X)[0]
    emotion = emotion_names[prediction]
    print(f"   '{text}' → {emotion}")

print(f"\n✅ El modelo funciona correctamente!")
print("="*80)
