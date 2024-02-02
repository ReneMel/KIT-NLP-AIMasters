import os
import pandas as pd

# Ruta al directorio que contiene los archivos CSV
directorio_csv = 'C:/Users/renemel/Documents/KIT/Datasets/CSV'

# Lista para almacenar los datos de cada CSV
datos_totales = []

# Diccionario para almacenar la cantidad de registros por tipo
conteo_tipos = {}

# Tipos de interés (modifica según tus necesidades)
tipos_interes = ['spam', 'ham']

# Mapeo de tipos a la categoría final
mapeo_tipos = {
    'not spam': 'ham',
    'Phishing': 'spam',
    'Commercial Spam': 'spam',
    'Phishing Email': 'spam',
    'Safe Email': 'ham'
}

for archivo_csv in os.listdir(directorio_csv):
    if archivo_csv.endswith('.csv'):
        # Leer el CSV y verificar si hay al menos una columna
        ruta_completa = os.path.join(directorio_csv, archivo_csv)
        try:
            datos_csv = pd.read_csv(ruta_completa)
        except pd.errors.EmptyDataError:
            print(f"El archivo {archivo_csv} está vacío. Se omitirá.")
            continue

        # Convertir los nombres de las columnas a minúsculas
        datos_csv.columns = map(str.lower, datos_csv.columns)

        # Comprobar si las columnas 'text' y 'type' existen en el DataFrame
        if 'text' in datos_csv.columns and 'type' in datos_csv.columns:
            # Seleccionar solo las columnas 'text' y 'type'
            datos_csv = datos_csv[['text', 'type']]

            # Mapear los tipos según las especificaciones
            datos_csv['type'] = datos_csv['type'].map(mapeo_tipos).fillna(datos_csv['type'])

            # Filtrar las filas donde 'type' está en la lista de tipos de interés
            datos_csv = datos_csv[datos_csv['type'].isin(tipos_interes)]

            # Agregar una nueva columna 'source' con el nombre del archivo
            datos_csv['source'] = archivo_csv

            # Agregar los datos del CSV a la lista
            datos_totales.append(datos_csv)

            # Actualizar el conteo de registros por tipo
            conteo_por_tipo = datos_csv['type'].value_counts().to_dict()
            for tipo, cantidad in conteo_por_tipo.items():
                conteo_tipos[tipo] = conteo_tipos.get(tipo, 0) + cantidad

# Concatenar todos los datos en un solo DataFrame
datos_finales = pd.concat(datos_totales, ignore_index=True)

# Mostrar el DataFrame final
print(datos_finales)

# Guardar el DataFrame final en un nuevo archivo CSV
datos_finales.to_csv('C:/Users/renemel/Documents/KIT/datos_combinados.csv', index=False)

# Mostrar la cantidad total de registros y el conteo por tipo
print(f'\nCantidad total de registros: {len(datos_finales)}')
print('\nConteo por tipo:')
for tipo, cantidad in conteo_tipos.items():
    print(f'{tipo}: {cantidad}')
