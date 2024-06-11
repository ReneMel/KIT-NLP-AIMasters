import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Cargar los datos desde un archivo CSV
df = pd.read_csv('../Data/combined-dataset.csv')

# Asumamos que la columna que contiene los textos se llama 'text'
textos = df['text'].tolist()

# Cargar el tokenizador y el modelo de análisis de sentimientos
tokenizador = AutoTokenizer.from_pretrained('roberta-base')
modelo = AutoModelForSequenceClassification.from_pretrained('roberta-base')
analizador_sentimientos = pipeline('sentiment-analysis', model=modelo, tokenizer=tokenizador)

# Función para dividir el texto en fragmentos de longitud máxima permitida por el modelo
def dividir_texto(texto, max_length=512):
    if not isinstance(texto, str) or not texto.strip():
        return []
    # Tokenizar el texto y obtener los IDs de los tokens
    encoding = tokenizador.encode_plus(texto, max_length=max_length, truncation=True)
    input_ids = encoding['input_ids']
    # Decodificar los input_ids a texto
    texto_tokenizado = tokenizador.decode(input_ids, skip_special_tokens=True)
    return texto_tokenizado



# Realizar el análisis de sentimientos en cada texto
resultados = []
for texto in textos:
    resultado = analizador_sentimientos(dividir_texto(texto))
    resultados.append(resultado)

# Convertir los resultados a un DataFrame
df_resultados = pd.DataFrame(resultados)

# Combinar los resultados con el DataFrame original
df['sentimiento'] = df_resultados['label']
df['score_sentimiento'] = df_resultados['score']

# Mostrar algunos resultados
print(df.head())

# Contar el número de sentimientos positivos y negativos
conteo_sentimientos = df_resultados['label'].value_counts()

# Calcular el porcentaje de cada sentimiento
total_sentimientos = len(df_resultados)
porcentaje_positivos = conteo_sentimientos.get('POSITIVE', 0) / total_sentimientos * 100
porcentaje_negativos = conteo_sentimientos.get('NEGATIVE', 0) / total_sentimientos * 100

print(f'Porcentaje de sentimientos positivos: {porcentaje_positivos:.2f}%')
print(f'Porcentaje de sentimientos negativos: {porcentaje_negativos:.2f}%')
