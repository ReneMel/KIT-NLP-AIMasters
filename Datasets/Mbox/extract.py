import mailbox
import csv
import os

def mbox_to_csv(mbox_file_path):
    # Crear un nombre de archivo CSV basado en el nombre del archivo Mbox
    csv_file = os.path.splitext(mbox_file_path)[0] + '.csv'

    mbox = mailbox.mbox(mbox_file_path)

    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['text', 'type']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterar sobre los mensajes en el archivo Mbox
        for message in mbox:
            # Obtener el cuerpo del correo
            body = message.get_payload()

            # Guardar en el archivo CSV con type: 'pishing' (puedes cambiarlo según tus necesidades)
            writer.writerow({'text': body, 'type': 'pishing'})

    print(f"Se ha creado el archivo CSV: {csv_file}")

# Reemplaza 'archivo.mbox' con la ruta completa de tu archivo Mbox
mbox_to_csv('C:/Users/renemel/Documents/KIT/Datasets/Mbox/phishing-2016.mbox')
