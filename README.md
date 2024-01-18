![Texto alternativo](assets/images/bannerAutomotriz.png)

# Proyecto Integrador: Machine Learning en investigación de mercado automotor
Proyecto integrador del módulo 6 Machine Learning de la carrera de Data Science en soyHenry

A un equipo de estudiantes del último módulo de la carrera de Ciencia de Datos se le ha encomendado la tarea de crear un modelo que le permita a una computadora aprender a partir de los datos suministrados y realizar predicciones sin la necesidad de programación explícita. Al finalizar el proyecto, dicho equipo deberá haber podido poner en prácticas las habilidades adquiridas a lo largo del módulo de machine learning (ML).

## Planteamiento de la problemática
Hemos sido contratados en el equipo de ciencias de datos en una consultora de renombre. Nos han asignado a un proyecto de estudio de mercado de una importante automotriz china. Nuestro cliente desea ingresar a nuestro mercado de automóviles, por lo que nos han encomendado analizar las características de los vehículos presentes en el mercado actual. Dado que tienen en su catálogo una amplia colección de modelos de todo tipo, cuyo catálogo está estratificado en gamas según el gusto de cada región, desean saber **qué características presentan los vehículos de gama alta y los de gama baja en nuestro mercado**, para poder **abarcar a todo los públicos objetivos** ajustándose a toda la demanda y, en base a estos datos, poder cotizar correctamente los vehículos que ofrecerá.

Nuestro _Data Lead_ nos ha recomendado que analicemos detalladamente los datos, los preprocesemos debidamente y que diseñemos **dos modelos predictivos**, uno para el precio y otro para distinguir vehículos de gama alta y de gama baja, utilizando la mediana de los precios como punto de corte. Desean obtener los archivos con las predicciones en formato de texto plano.

Además del análisis detallado de la exploración de los datos, estas son las dos predicciones posibles que les interesaría analizar utilizando los datos que se han puesto a su disposición:
1. Implementar un modelo de clasificación con aprendizaje supervisado que permita clasificar el precio de los vehículos en baratos y caros usando la mediana de los precios como punto de corte.
2. Implementar un modelo de regresión con aprendizaje supervisado que permita predecir el precio final de los vehículos.

### CRISP-DM
La metodología Proceso Estándar de Toda la Industria para Minería de Datos (CRISP-DM en inglés) es utilizada en proyectos de análisis de datos y minería de datos. Se puede considerar como la metodología estándar en la industria para proyectos dedicados a extraer valor de los datos. Estaremos resumiendo sus fases dentro del proyecto en:

- Fase 1: Análisis exploratorio de datos
  - Comprensión del negocio
  - Comprensión de los datos

- Fase 2: Preparación de datos
  - Preparación de los datos
          
- Fase 3: Modelamiento y evaluación
  - Modelado
  - Evaluación

La última fase de CRISP-DM, la de despliegue, queda fuera del alcance y objetivos del proyecto debido a que implementa el modelo en un entorno de producción y se realiza un seguimiento continuo para asegurar su correcto funcionamiento.


## 1 Exploremos los datos
### Comprensión del negocio
El mercado automotor está muy ligado a la cultura de cada país según los gustos de cada uno, por ejemplo:
- El mercado norteamericano valora mucho los motores y vehículos muy grandes.
- El mercado europeo prefiere el bajo consumo.
- El mercado latinoamericano, los precios finales bajos.

y así varía según la región, elpaís, el nivel socioeconómico o la cultura.<br>

Un mismo vehículo puede tener un valor muy distinto de un país al otro, y no solo por los impuestos o costos de producción, sino por cómo cotiza el modelo en el mercado.

### Comprensión de los datos

Para ello, nuestro departamento de datos ha recopilado precios y características de varios de los modelos de vehículos disponibles en nuestro mercado, junto con sus precios de venta al público. Y han armado el siguiente diccionario de datos:
<details open><summary>Diccionario</summary>
<pre>
   NOMBRE DE LA VARIABLE   TIPO      DESCRIPCIÓN
   -----------------------------------------------------------------------------------------------
   car_ID                  Int       Número de Identificación del vehículo en la base de datos
   symboling               Int       Calificación de riesgo asociada al vehículo, +3 es riesgoso
                                     poco seguro, -3 es poco riesgoso muy seguro
   CarName                 Str       Nombre de fantasía del vehículo
   fueltype                Str       Tipo de combustible
   aspiration              Str       Tipo de aspiración del motor
   doornumber              Str       Número de puertas
   carbody                 Str       Tipo de carrocería del vehículo
   drivewheel              Str       Ubicación del volante del conductor
   enginelocation          Str       Ubicación del motor en el vehículo
   wheelbase               Float     Distancia entre ejes
   carlength               Float     Longitud del vehículo
   carwidth                Float     Ancho del vehículo
   carheight               Float     Altura del vehículo
   curbweight              Int       Peso del vehículo sin carga ni ocupantes
   enginetype              Str       Tipo de motor
   cylindernumber          Str       Número de cilindros del motor
   enginesize              Int       Tamaño del motor
   fuelsystem              Str       Sistema de administración de combustible del motor
   boreratio               Float     Relación diámetro/carrera de los pistones del motor
   stroke                  Float     Volumen de cilindrada
   compressionratio        Float     Relación de compresión del aire dentro del motor
   horsepower              Int       Potencia del vehículo, en caballos de fuerza (HP)
   peakrpm                 Int       Revoluciones máximas que soporta el motor
   citympg                 Int       Consumo en ciudad, en millas por galón de combustible
   highwaympg              Int       Consumo en ruta, en millas por galón de combustible
   price                   Float     Precio del vehículo
</pre>
</details>


