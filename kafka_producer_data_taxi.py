import json
import time
import random
from kafka import KafkaProducer
from pyspark.sql import SparkSession

# Inicializa la sesión de Spark
spark = SparkSession.builder \
    .appName("KafkaProducerFromParquet") \
    .getOrCreate()

# Cargar el archivo Parquet desde HDFS
input_path = "hdfs://localhost:9000/data_clean/df_taxis.parquet"
df_taxis = spark.read.parquet(input_path)

# Configura el productor de Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Extrae registros de `df_taxis` y envíalos a Kafka
while True:
    # Selecciona un registro aleatorio
    row = df_taxis.sample(fraction=0.001).limit(1).collect()  # Ajusta el 'fraction' si deseas más o menos registros aleatorios
    
    if row:
        record = row[0].asDict()  # Convierte el registro en un diccionario

        # Serializa los datos relevantes
        taxi_data = {
            "VendorID": record["VendorID"],
            "pickup_datetime": record["pickup_datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "dropoff_datetime": record["dropoff_datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "PULocationID": record["PULocationID"],
            "DOLocationID": record["DOLocationID"],
            "passenger_count": record["passenger_count"],
            "trip_distance": record["trip_distance"],
            "fare_amount": record["fare_amount"],
            "extra": record["extra"],
            "mta_tax": record["mta_tax"],
            "tip_amount": record["tip_amount"],
            "tolls_amount": record["tolls_amount"],
            "ehail_fee": record["ehail_fee"],
            "improvement_surcharge": record["improvement_surcharge"],
            "total_amount": record["total_amount"],
            "payment_type": record["payment_type"],
            "congestion_surcharge": record["congestion_surcharge"],
            "Airport_fee": record["Airport_fee"],
            "taxi_type": record["taxi_type"],
            "tip_amount_is_null": record["tip_amount_is_null"],
            "passenger_count_is_null": record["passenger_count_is_null"]
        }
        
        # Enviar el registro al topic de Kafka
        producer.send('taxi_data', value=taxi_data)
        print(f"Sent: {taxi_data}")

        # Espera un intervalo de tiempo variable
        time.sleep(random.uniform(0.5, 2))

# Finalizar la sesión de Spark 
# spark.stop()