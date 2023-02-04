"""
Authors: Sonia Castro Paniello
         Olga Villar Cairó
Data: 11/01/23
"""

"""
RUN-TIME CLASSIFIER PIPELINE STEPS:
    (1) Read the aircraft and day that will constitute the new record.
    (2) Search and add the KPI values for that aircraft and day in the DW database.
    (3) Construct the sensor dataframe as done in the first pipeline but adding a filter to get only the average for the input record.
    (4) Create the complete new record.
    (5) Classify this record using the trainned model.
"""

import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, DateType, StringType, DoubleType
from datetime import datetime
from easyinput import read
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

if(__name__== "__main__"):
    #creates spark session
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("BDAlab") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

#establishes connection with the table aircraftutilization from DW
table_DW = (spark.read \
.format("jdbc") \
.option("driver","org.postgresql.Driver") \
.option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
.option("dbtable", "public.aircraftutilization") \
.option("user", "sonia.castro") \
.option("password", "DB210402") \
.load())

#STEP (1)
#reads the input (aircraft and day) that will be predicted
print('Introduce the aircraft ID')
inputaircraft=read(str)
print('Introduce the data in the format yyyy-mm-dd')
inputdate=read(str)
inputdate = datetime.strptime(inputdate, '%Y-%m-%d').date()

#STEP (2)
#gets the KPI values for the input aircraft and day
inputrow = table_DW.select("timeid","aircraftid","flighthours", "flightcycles","delayedminutes")\
.filter((f.col("timeid") == inputdate) & (f.col("aircraftid") == inputaircraft)).cache()

#STEP (3)
#gets the average of the sensor for the input aircraft and day (similar to step (1) from the first pipeline)
schema0 = StructType([
    StructField("timeid", DateType(), True),
    StructField("aircraftid", StringType(), True),
    StructField("Sensor_avg", DoubleType(), True)
])

#computes all the sensor averages grouped by aircraft and date
sensor = (sc.wholeTextFiles("./resources/trainingData"))\
.map(lambda x: ("-".join(x[0].split("/")[9][:-4].split("-")[-2:]), x[1].split("\n")[1:-1]))\
.flatMap(lambda x: [(x[0], y) for y in x[1]])\
.map(lambda x:(((datetime.strptime(x[1].split(";")[0].split(" ")[0], '%Y-%m-%d').date() , x[0]), x[1].split(";")[2])))\
.mapValues(lambda x: (x,1.0))\
.reduceByKey(lambda x1,x2: (float(x1[0])+ float(x2[0]), x1[1]+x2[1]))\
.mapValues(lambda x: (float(x[0])/x[1]))\
.map(lambda x: (x[0][0], x[0][1], x[1]))
#creates a one-row dataframe with the average, date and aircraft for the input values
df_sensors = spark.createDataFrame(sensor, schema0)\
.filter((f.col("timeid") == inputdate)  & (f.col("aircraftid") == inputaircraft)).cache()


if df_sensors.count()==0 or inputrow.count() == 0:
    print('ERROR! There is no record for the aircraft ', inputaircraft, ' on ', inputdate)

else:
    #STEP (4): creates the completed record
    inputrow = inputrow.join(df_sensors, ["timeid", "aircraftid"])\
    .drop("timeid", "aircraftid")\
    .select(f.col("Sensor_Avg"), f.col("flighthours"), f.col( "flightcycles"), f.col("delayedminutes"))

    #STEP (5): imports the model and does the prediction
    model = CrossValidatorModel.load("/mnt/c/users/sonia/desktop/CodeSkeleton/model.bin")
    assembler = VectorAssembler(inputCols=inputrow.columns, outputCol='features')
    sample = assembler.transform(inputrow)
    prediction = model.transform(sample)
    if prediction.select('prediction').first()[0] == 1.0:
        print("Unscheduled maintenance predicted in the next 7 days for that flight.")
    else:
        print("No maintenance predicted in the next 7 days for that flight.")
