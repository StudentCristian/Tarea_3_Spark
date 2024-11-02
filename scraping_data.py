import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from hdfs import InsecureClient

def obtener_urls_filtradas(url):
    """
    Obtiene las URLs filtradas de la página web https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
    """
    filtered_urls = []
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            absolute_url = urljoin(url, href)
            if any(keyword in absolute_url for keyword in ["yellow_tripdata", "green_tripdata"]) and "2023" in absolute_url:
                if any(month in absolute_url for month in ["-01", "-04", "-07", "-10"]):
                    filtered_urls.append(absolute_url)
    return filtered_urls

def process_and_save_trip_data(filtered_urls):
    """
    Descarga cada URL usando wget y lo guarda en HDFS.
    """
    # Configurar el cliente HDFS
    hdfs_client = InsecureClient('http://localhost:9870', user='hadoop')

    for url in filtered_urls:
        try:
            # Generar nombre de archivo
            parquet_file_name = os.path.basename(url).strip()

            # Descargar el archivo usando wget
            local_path = f'/tmp/{parquet_file_name}'
            os.system(f'wget -O {local_path} {url}')
            print(f"Archivo descargado: {local_path}")

            # Verificar que el archivo se haya descargado correctamente
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"El archivo {local_path} no se encontró después de la descarga.")

            # Crear el path en HDFS
            hdfs_path = f'/datasets/{parquet_file_name}'

            # Subir archivo a HDFS usando el cliente HDFS
            with open(local_path, 'rb') as local_file:
                hdfs_client.write(hdfs_path, local_file)

            print(f"Archivo subido a HDFS: {hdfs_path}")

            # Limpiar archivo temporal
            os.remove(local_path)
            print(f"Archivo temporal eliminado: {local_path}")

        except Exception as e:
            print(f"Error al procesar {url}: {e}")

url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
filtered_urls = obtener_urls_filtradas(url)
process_and_save_trip_data(filtered_urls)