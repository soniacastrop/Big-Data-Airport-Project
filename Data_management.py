"""
Authors: Sonia Castro Paniello
         Olga Villar Cairó
Data: 11/01/23
"""

"""
DATA MANAGEMENT PIPELINE STEPS:
    (1) Get the sensor data that contains an average value of the sensor values for each aircraft and day using an RDD to parallelize.
    (2) Get the KPIs for each aircraft and day from the DW database.
    (3) Join the sensor data with the KPIs to obtain the samples.
    (4) Get the interruptions reported by the sensor 3453 from the AMOS database.
    (5) Add a label to the samples using 1 if the aircraft needed maintenance the given day and 0 if not.
    (6) Adapt the dataframe to predict one week in advance 1 by changing the label of the rows that correspond to the six days before the maintenance took place.
"""

import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, DateType, StringType, DoubleType
from datetime import datetime, timedelta


HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"


def Get_sensor_data(spark,sc):
    '''
    Corresponds to step (1)
    Gets all the data contained in the trainingData directory into an RDD. Extracts the aircraft from the filepath and divides the content of each csv in rows.
    With a flatMap we convert each row of the rdd to be (aircraft, row). From each row we extract the date and the sensor value and map or rdd to have
    the key (date,aircraft) and the value (sensor value, 1).Then, the average of the sensor by key is computed by a reduceByKey dividing the sum of all sensor values
    by the number of "1". Finally, this informstion is stored in a dataframe, which has <timeid> <aircraftif> and <Sensor_avg> as columns.
    Returns the resulting dataframe, which has key <timeid, aircraftid>.
    '''
    print("Obtaining data from the flight sensors...")

    #creates the schema of the final dataframe
    schema0 = StructType([
        StructField("timeid", DateType(), True),
        StructField("aircraftid", StringType(), True),
        StructField("Sensor_avg", DoubleType(), True)
    ])

    #gets the average and structures the data as in the previous schema
    sensor = (sc.wholeTextFiles("./resources/trainingData"))\
    .map(lambda x: ("-".join(x[0].split("/")[9][:-4].split("-")[-2:]), x[1].split("\n")[1:-1]))\
    .flatMap(lambda x: [(x[0], y) for y in x[1]])\
    .map(lambda x:(((datetime.strptime(x[1].split(";")[0].split(" ")[0], '%Y-%m-%d').date() , x[0]), x[1].split(";")[2])))\
    .mapValues(lambda x: (x,1.0))\
    .reduceByKey(lambda x1,x2: (float(x1[0])+ float(x2[0]), x1[1]+x2[1]))\
    .mapValues(lambda x: (float(x[0])/x[1]))\
    .map(lambda x: (x[0][0], x[0][1], x[1]))
    df_sensors = spark.createDataFrame(sensor, schema0).cache()
    return df_sensors


def Get_DW_data(spark):
    '''
    Corresponds to step (2)
    Establishes the connection with the "aircraftutilization" table from the DW database. From which selects the required columns
    ("timeid", "aircraftid", "flighthours", "flightcycles" and "delayedminutes") and returns a dataframe with the content.
    '''
    print("Obtaining data from DW Aircraftutilization table...")

    table_DW = (spark.read \
    .format("jdbc") \
    .option("driver","org.postgresql.Driver") \
    .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
    .option("dbtable", "public.aircraftutilization") \
    .option("user", "sonia.castro") \
    .option("password", "********") \
    .load()).select("timeid","aircraftid","flighthours", "flightcycles","delayedminutes").cache()
     #password not shown
    return table_DW


def Get_AMOS_data(spark):
    '''
    Corresponds to step (4)
    Establishes the connection with the "operationinterruption" table from the AMOS database. From which selects "starttime" as "timeid"
    and "aircraftregistration" as "aircraftid". It also filters the flight interruptions to make sure they were reported by subsystem 3453.
    Returns a dataframe with the previous information.
    '''
    print("Obtaining data from AMOS operationinterruption table...")
    table_AMOS = (spark.read \
    .format("jdbc") \
    .option("driver","org.postgresql.Driver") \
    .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
    .option("dbtable", "oldinstance.operationinterruption") \
    .option("user", "sonia.castro") \
    .option("password", "********") \
    .load())
    #password not shown
         
    df_AMOS = table_AMOS.select(f.date_format(table_AMOS.starttime, 'yyyy-MM-dd').alias('timeid'), (table_AMOS.aircraftregistration).alias("aircraftid"), "subsystem")\
    .filter(f.col("subsystem")=="3453")\
    .drop("subsystem").cache()

    return df_AMOS


def Create_labeled_data(spark, sc):
    '''
    Corresponds to steps (1) (2) (3) (4) (5)
    Gets two processed dataframes, one with the sensors data and another with the kpis. Joins them using time and aircraft (primary key), getting
    all the flight information in one dataframe. Then, gets the interruption table and adds a column corresponding to the label, which in this case equals 1
    because the flights of this table needed maintenance. To complete the labels, joins both tables using an outer left join in order to give null value to
    the label of the rows that only exist in the first table (as they did not need maintenance) and 1 if they exist in both tables. Finally, returns the final
    dataset by adding the complete label column, which has value 1 if the aircraft had maintenance that day and 0 if it did not.
    '''
    print("Creating labeled data...")

    df_sensors=Get_sensor_data(spark,sc) #STEP (1)
    df_dw = Get_DW_data(spark) #STEP (2)
    df_unlabeled = df_sensors.join(df_dw, ["timeid", "aircraftid"]) #STEP (3): table with all prediction features

    df_AMOS = Get_AMOS_data(spark) #STEP (4)
    df_AMOS = df_AMOS.withColumn('label_fake', f.lit(1)) #table with the interruptions and a temporary label column

    #gets labeled data and drops the temporary label column
    df_labeled = df_unlabeled.join(df_AMOS, ['timeid', 'aircraftid'], 'left')
    df_labeled = df_labeled.withColumn('label', f.when(df_labeled['label_fake'].isNotNull(), 1).otherwise(0))\
    .drop("label_fake") #STEP (5)

    return df_labeled


def Change_7days_before(spark, df_labeled):
    '''
    Corresponds to step (6)
    Adapts the dataframe taking into account that the prediction needs to give information about the next 7 days. To do so, for each sample with label 1 (maintenance),
    another 6 samples with the same label are created for the 6 previous days and added in 'modified_rdd'. These fabricated rows have the same values as the original.
    Then, the initial dataset is left outer joinned with the new rows and a map function is applied so that only the rows that exist in both datasets have their label
    modified.  So that the rows that were fabricated but did not originally exist are not added to the dataset and the ones with label 0 that don't appear in the new
    dataset are not modified.
    In the end returns the labeled dataframe adapted to make predictions one week in advance.
    '''
    schema1 = StructType([
        StructField("timeid", DateType(), True),
        StructField("aircraftid", StringType(), True),
        StructField("Sensor_avg", DoubleType(), True),
        StructField("flighthours", DoubleType(), True),
        StructField("flightcycles", DoubleType(), True),
        StructField("delayedminutes", DoubleType(), True),
        StructField("label", DoubleType(), True),
    ])

    #creates a new RDD
    modified_rdd = df_labeled.where(df_labeled['label'] == 1).rdd
    def modify_row(row):
        '''
        Given a row, returns a list of 6 rows with the same values as the input row except for the date.
        Each new row's date corresponds to a day of the week before the original date.
        '''
        current_date = row['timeid']
        date_list = [current_date - timedelta(days=x) for x in range(6, 0, -1)]
        new_rows = []
        for date in date_list:
            #creates a new row with the same values as the original row with the date replaced, the same aircraft and the rest of columns as floats
            values = list(row)
            values[0] = date
            values[2] = float(values[2])
            values[3] = float(values[3])
            values[4] = float(values[4])
            values[5] = float(values[5])
            values[6] = float(values[6])
            new_row = Row(*values)
            new_rows.append(new_row)
        return new_rows

    modified_rdd = modified_rdd.flatMap(modify_row)
    #stores the dataframe with the previously defined schema
    new_rows = spark.createDataFrame(modified_rdd, schema1).cache()
    new_rows = new_rows.dropDuplicates(["timeid", "aircraftid"]) # por que igual si se llevan menos d una semana y estan con 1 saldran varias veces y en el join lo joden


    #changes to RDD structure to be able to parallelize
    new_rows =new_rows.rdd.map(lambda row: ((row.timeid, row.aircraftid), row))
    df_labeled = df_labeled.rdd.map(lambda row: ((row.timeid, row.aircraftid), row))
    #handles the exception case by joining the dataset with the new rows and all the original labeled data. If a row only appears in the original data, it is not modified
    #and if it appears in both, the label is replaced with 1.
    df_final = df_labeled.leftOuterJoin(new_rows)
    df_final = df_final.map(lambda row: (row[0][0], row[0][1],float(row[1][0].Sensor_avg), float(row[1][0].flighthours), float(row[1][0].flightcycles), \
    float(row[1][0].delayedminutes), float(row[1][1].label)) if row[1][1] is not None else (row[0][0], \
    row[0][1],float(row[1][0].Sensor_avg),float(row[1][0].flighthours), float(row[1][0].flightcycles), float(row[1][0].delayedminutes), \
    float(row[1][0].label)))\
    .toDF(schema1)
    return df_final

def main():
    #configures and establishes spark session
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("BDAlab") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()
    #creates the dataframe of all labeled data
    df_labeled = Create_labeled_data(spark, sc)
    #adapts the dataframe to predict one week in advance
    df_final = Change_7days_before(spark, df_labeled)
    #saves prediction data
    print("Trainning data stored.")
    df_final.write.mode('overwrite').save('labeled_data.parquet', format='parquet')

main()
