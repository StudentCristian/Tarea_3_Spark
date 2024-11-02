
### Diccionario de datos de taxis Green

| **Nombre del campo**         | **Descripción**                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------|
| **VendorID**                 | Código que indica el proveedor de LPEP que generó el registro. **1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.** |
| **lpep_pickup_datetime**     | Fecha y hora en la que el taxímetro fue activado.                                                           |
| **lpep_dropoff_datetime**    | Fecha y hora en la que el taxímetro fue desactivado.                                                        |
| **store_and_fwd_flag**       | Indica si el registro del viaje se almacenó en la memoria del vehículo antes de enviarlo al proveedor, debido a la falta de conexión con el servidor. **Y= viaje almacenado; N= viaje no almacenado** |
| **RatecodeID**               | Código de tarifa final aplicada al final del viaje. **1= Tarifa estándar; 2= JFK; 3= Newark; 4= Nassau o Westchester; 5= Tarifa negociada; 6= Viaje en grupo** |
| **PULocationID**             | Código de la zona TLC donde se inició el viaje.                                                             |
| **DOLocationID**             | Código de la zona TLC donde finalizó el viaje.                                                              |
| **passenger_count**          | Número de pasajeros en el vehículo. Este valor es ingresado por el conductor.                               |
| **trip_distance**            | Distancia recorrida en el viaje en millas, reportada por el taxímetro.                                      |
| **fare_amount**              | Tarifa calculada por el taxímetro basada en tiempo y distancia.                                             |
| **extra**                    | Extras y recargos misceláneos, incluyendo el recargo de $0.50 por hora pico y de $1 durante la noche.       |
| **mta_tax**                  | Impuesto de $0.50 para la MTA, que se activa automáticamente según la tarifa en uso.                        |
| **tip_amount**               | Monto de la propina. Este campo se completa automáticamente para propinas con tarjeta de crédito; no incluye propinas en efectivo. |
| **tolls_amount**             | Monto total de peajes pagados durante el viaje.                                                             |
| **ehail_fee**                | Tarifa adicional para las llamadas electrónicas.                                                            |
| **improvement_surcharge**    | Recargo de mejora de $0.30 aplicado en viajes a pedido en la parada de taxis.                              |
| **total_amount**             | Monto total cobrado al pasajero. No incluye propinas en efectivo.                                           |
| **payment_type**             | Código numérico que indica cómo el pasajero pagó el viaje. **1= Tarjeta de crédito; 2= Efectivo; 3= Sin cargo; 4= Disputa; 5= Desconocido; 6= Viaje anulado** |
| **trip_type**                | Código que indica si el viaje fue una solicitud en la calle o una asignación automática del proveedor. **1= Solicitud en la calle; 2= Despacho** |
| **congestion_surcharge**     | Monto recaudado en el viaje por el recargo de congestión de Nueva York.                                     |

### Diccionario de datos de taxis Yellow

| **Nombre del campo**         | **Descripción**                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------|
| **VendorID**                 | Código que indica el proveedor de TPEP que generó el registro. **1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.** |
| **tpep_pickup_datetime**     | Fecha y hora en la que el taxímetro fue activado.                                                           |
| **tpep_dropoff_datetime**    | Fecha y hora en la que el taxímetro fue desactivado.                                                        |
| **passenger_count**          | Número de pasajeros en el vehículo. Este valor es ingresado por el conductor.                               |
| **trip_distance**            | Distancia recorrida en el viaje en millas, reportada por el taxímetro.                                      |
| **RatecodeID**               | Código de tarifa final aplicada al final del viaje. **1= Tarifa estándar; 2= JFK; 3= Newark; 4= Nassau o Westchester; 5= Tarifa negociada; 6= Viaje en grupo** |
| **store_and_fwd_flag**       | Indica si el registro del viaje se almacenó en la memoria del vehículo antes de enviarlo al proveedor, debido a la falta de conexión con el servidor. **Y= viaje almacenado; N= viaje no almacenado** |
| **PULocationID**             | Código de la zona TLC donde se inició el viaje.                                                             |
| **DOLocationID**             | Código de la zona TLC donde finalizó el viaje.                                                              |
| **payment_type**             | Código numérico que indica cómo el pasajero pagó el viaje. **1= Tarjeta de crédito; 2= Efectivo; 3= Sin cargo; 4= Disputa; 5= Desconocido; 6= Viaje anulado** |
| **fare_amount**              | Tarifa calculada por el taxímetro basada en tiempo y distancia.                                             |
| **extra**                    | Extras y recargos misceláneos, incluyendo el recargo de $0.50 por hora pico y de $1 durante la noche.       |
| **mta_tax**                  | Impuesto de $0.50 para la MTA, que se activa automáticamente según la tarifa en uso.                        |
| **tip_amount**               | Monto de la propina. Este campo se completa automáticamente para propinas con tarjeta de crédito; no incluye propinas en efectivo. |
| **tolls_amount**             | Monto total de peajes pagados durante el viaje.                                                             |
| **improvement_surcharge**    | Recargo de mejora de $0.30 aplicado en viajes a pedido en la parada de taxis.                              |
| **total_amount**             | Monto total cobrado al pasajero. No incluye propinas en efectivo.                                           |
| **congestion_surcharge**     | Monto recaudado en el viaje por el recargo de congestión de Nueva York.                                     |
| **airport_fee**              | Tarifa de $1.25 aplicada solo en recogidas en los aeropuertos de LaGuardia y John F. Kennedy.              |

