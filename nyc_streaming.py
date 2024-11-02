from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, hour
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# Configura la sesión de Spark
spark = SparkSession.builder \
    .appName("TaxiDataConsumer") \
    .getOrCreate()

# Establecer el nivel de log a WARN
spark.sparkContext.setLogLevel("WARN")

# Define el esquema de los datos
schema = StructType([
    StructField("VendorID", IntegerType()),
    StructField("pickup_datetime", TimestampType()),
    StructField("dropoff_datetime", TimestampType()),
    StructField("PULocationID", IntegerType()),
    StructField("DOLocationID", IntegerType()),
    StructField("passenger_count", DoubleType()),
    StructField("trip_distance", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("extra", DoubleType()),
    StructField("mta_tax", DoubleType()),
    StructField("tip_amount", DoubleType()),
    StructField("tolls_amount", DoubleType()),
    StructField("ehail_fee", DoubleType()),
    StructField("improvement_surcharge", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("payment_type", IntegerType()),
    StructField("congestion_surcharge", DoubleType()),
    StructField("Airport_fee", DoubleType()),
    StructField("taxi_type", StringType()),
    StructField("tip_amount_is_null", IntegerType()),
    StructField("passenger_count_is_null", IntegerType())
])

# Lee los datos de Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "taxi_data") \
    .load()

# Parsear los datos de JSON a columnas
parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Análisis de ingresos y pasajeros
revenue_passenger_df = parsed_df.groupBy(
    window("pickup_datetime", "15 minutes"), "PULocationID"
).agg(
    {"fare_amount": "sum", "passenger_count": "sum"}
).withColumnRenamed("sum(fare_amount)", "total_revenue") \
 .withColumnRenamed("sum(passenger_count)", "total_passengers")

# Análisis de conteo de viajes
trip_count_df = parsed_df.groupBy(
    window("pickup_datetime", "15 minutes"), "PULocationID"
).agg(
    {"payment_type": "count"}
).withColumnRenamed("count(payment_type)", "total_trips")

# Análisis de distancia promedio
avg_distance_df = parsed_df.groupBy(
    window("pickup_datetime", "15 minutes"), "PULocationID"
).agg(
    {"trip_distance": "avg"}
).withColumnRenamed("avg(trip_distance)", "avg_trip_distance")

# Análisis de conteo de viajes por tipo de pago
payment_type_count_df = parsed_df.groupBy("payment_type").agg(
    {"payment_type": "count"}
).withColumnRenamed("count(payment_type)", "total_trips")

# Análisis de distancia promedio por ubicación
avg_distance_location_df = parsed_df.groupBy("PULocationID").agg(
    {"trip_distance": "avg"}
).withColumnRenamed("avg(trip_distance)", "avg_trip_distance")

# Análisis de distancia promedio por hora
avg_distance_hour_df = parsed_df.groupBy(
    hour("pickup_datetime").alias("hour")
).agg(
    {"trip_distance": "avg"}
).withColumnRenamed("avg(trip_distance)", "avg_trip_distance")

# Muestra el resultado de ingresos y pasajeros en la consola
query_revenue_passenger = revenue_passenger_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime='30 seconds') \
    .start()

# Muestra el resultado de conteo de viajes en la consola
query_trip_count = trip_count_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime='30 seconds') \
    .start()

# Muestra el resultado de distancia promedio en la consola
query_avg_distance = avg_distance_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime='30 seconds') \
    .start()

# Muestra el resultado de conteo de viajes por tipo de pago
query_payment_type = payment_type_count_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime='30 seconds') \
    .start()

# Muestra el resultado de distancia promedio por ubicación
query_avg_distance_location = avg_distance_location_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime='30 seconds') \
    .start()

# Muestra el resultado de distancia promedio por hora
query_avg_distance_hour = avg_distance_hour_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .trigger(processingTime='30 seconds') \
    .start()

# Espera a que todas las consultas terminen
query_revenue_passenger.awaitTermination()
query_trip_count.awaitTermination()
query_avg_distance.awaitTermination()
query_payment_type.awaitTermination()
query_avg_distance_location.awaitTermination()
query_avg_distance_hour.awaitTermination()