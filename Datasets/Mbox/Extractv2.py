import mailbox
import csv
import os
import re

def getcharsets(msg):
    """
    Obtiene el conjunto de charsets utilizados en el mensaje.
    """
    charsets = set()
    for c in msg.get_charsets():
        if c is not None:
            charsets.add(c)
    return charsets

def handleerror(errmsg, emailmsg, charset):
    """
    Maneja errores de decodificación e imprime información relevante.
    """
    print()
    print(errmsg)
    print(f"This error occurred while decoding with {charset} charset.")
    print("These charsets were found in the one email:", getcharsets(emailmsg))
    print("This is the subject:", emailmsg['subject'])
    print("This is the sender:", emailmsg['From'])

def getbodyfromemail(msg):
    """
    Extrae el cuerpo de texto del mensaje de correo electrónico.
    """
    body = None
    
    # Recorrer las partes del mensaje para encontrar el cuerpo de texto.
    if msg.is_multipart():
        for part in msg.walk():
            # Si la parte es multipart, recorrer las subpartes.
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    elif msg.get_content_type() == 'text/plain':
        body = msg.get_payload(decode=True)

    # Intentar decodificar el cuerpo utilizando los charsets encontrados.
    if body is not None:
        for charset in getcharsets(msg):
            try:
                body = body.decode(charset)
                break  # Decodificación exitosa, salir del bucle.
            except UnicodeDecodeError:
                handleerror("UnicodeDecodeError: encountered.", msg, charset)
            except AttributeError:
                handleerror("AttributeError: encountered.", msg, charset)
    return body

def extract_urls(text):
    """
    Extrae todas las URLs del texto del cuerpo del correo electrónico.
    """
    url_pattern = re.compile(r'http[s]?://\S+')
    urls = url_pattern.findall(text)
    return urls

# Inicializar contador de correos con cuerpo encontrado
total_emails_with_body = 0

# Nombre del archivo CSV
csv_filename = 'private-phishing-2018.csv'

# Determinar si el archivo ya existe
file_exists = os.path.isfile(csv_filename)

# Abrir archivo CSV en modo apéndice para agregar datos
with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['sender', 'receiver', 'date', 'subject', 'body', 'urls']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Escribir la cabecera solo si el archivo no existe
    if not file_exists:
        writer.writeheader()

    # Procesar cada archivo mbox
    for year in range(2024, 2026):
        mboxfile = f'data/phishing-{year}.mbox'
        print(f"Processing {mboxfile}")
        
        try:
            for thisemail in mailbox.mbox(mboxfile):
                body = getbodyfromemail(thisemail)
                if body:
                    total_emails_with_body += 1
                    sender = thisemail['From'] if thisemail['From'] else "No sender"
                    receiver = thisemail['To'] if thisemail['To'] else "No receiver"
                    date = thisemail['Date'] if thisemail['Date'] else "No date"
                    subject = thisemail['subject'] if thisemail['subject'] else "No subject"
                    # Decodificar el cuerpo si está en bytes
                    if isinstance(body, bytes):
                        body = body.decode('utf-8', errors='replace')
                    urls = extract_urls(body)
                    urls_str = ', '.join(urls)
                    writer.writerow({'sender': sender, 'receiver': receiver, 'date': date, 'subject': subject, 'body': body, 'urls': urls_str})
                else:
                    print("No body found in this email.")
        except FileNotFoundError:
            print(f"{mboxfile} not found, skipping.")
        except Exception as e:
            print(f"An error occurred while processing {mboxfile}: {e}")

# Imprimir cantidad total de correos con cuerpo encontrado
print(f"Total emails with body found: {total_emails_with_body}")
