# Databricks notebook source
# Install Unity Catalog AI integration packages with the Databricks extra
%pip install unitycatalog-ai[databricks]==0.3.2
%pip install mlflow-skinny[databricks]==3.7.0
dbutils.library.restartPython()

# COMMAND ----------

# Import the required libraries 
import mlflow
import json

# COMMAND ----------

def process_csv(databricks_share_name: str):
    # Read the CSV file from the volume with headers
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("multiLine", "true") \
        .option("escape", '"') \
        .load(f"/Volumes/{databricks_share_name}/v01/sf-listings/sf-airbnb.csv")

    # Check the schema and first few rows
    print("Schema:")
    df.printSchema()

    print("\nRow count:")
    print(df.count())

    print("\nSample data:")
    display(df.limit(5))

    # Write as a Delta table
    df.write.format("delta") \
        .mode("overwrite") \
        .saveAsTable("sf_airbnb_listings")

    print("\nDelta table created successfully!")
    return df