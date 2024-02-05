import mailbox
import pandas as pd
import os

# Ruta al directorio que contiene los archivos mbox
directorio_mbox = 'C:/Users/renemel/Documents/KIT/Datasets/Mbox'

# Lista para almacenar los datos de cada correo
datos_correos = []

# Tipo de correo para esta tarea (en este caso, "phishing")
tipo_correo = "phishing"

# Iterar sobre cada archivo mbox en el directorio
for archivo_mbox in os.listdir(directorio_mbox):
    if archivo_mbox.endswith('.mbox'):
        ruta_completa = os.path.join(directorio_mbox, archivo_mbox)

        # Leer el archivo mbox
        mbox = mailbox.mbox(ruta_completa)

        # Iterar sobre cada correo en el archivo mbox
        for mensaje in mbox:
            # Obtener el cuerpo del correo
            cuerpo_correo = mensaje.get_payload()

            # Agregar los datos del correo a la lista
            datos_correos.append({
                'text': cuerpo_correo,
                'type': tipo_correo
            })

# Crear un DataFrame de pandas con los datos de los correos
df_correos = pd.DataFrame(datos_correos)

# Mostrar el DataFrame
print(df_correos)

# Guardar el DataFrame en un archivo CSV
df_correos.to_csv('/ruta/al/directorio/correos_phishing.csv', index=False)
