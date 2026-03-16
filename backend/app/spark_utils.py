from pyspark.sql import SparkSession

def get_spark_session():
    spark = SparkSession.builder \
        .appName("FinPulseETL") \
        .master("local[*]") \
        .getOrCreate()
    
    return spark