# Databricks notebook source
# MAGIC %run ./Classroom-Setup-Common

# COMMAND ----------

!pip install dspy==3.0.4
!pip install mlflow==3.4.0 
!pip install "pydantic>=2.0.0,<3.0.0"
dbutils.library.restartPython()

# COMMAND ----------

import os

def dev_lab_setup(catalog_name, schema_name=None):
    if schema_name is None:
        schema_name=os.path.basename(os.getcwd())
        schema_name=schema_name.replace('-','_')
    spark.sql(f"USE CATALOG {catalog_name}")
    print(f"Using catalog: {catalog_name}")
    try:
        spark.sql(f"USE SCHEMA {schema_name}")
        print(f"Using schema: {schema_name}")
    except: 
        print(f"Schema {schema_name} does not exist. Creating it.")
        spark.sql(f"CREATE SCHEMA {schema_name}")
        spark.sql(f"USE SCHEMA {schema_name}")
        print(f"Schema {schema_name} created.\nUsing schema: {schema_name}")
    return None

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

# COMMAND ----------

def tool_creation(catalog_name:str, schema_name: str):
    catalog_query = f"USE CATALOG {catalog_name}"
    schema_query = f"USE SCHEMA {schema_name}"
    tool1 = """
            CREATE OR REPLACE FUNCTION avg_neigh_price(
            neighborhood_name STRING COMMENT "The neighborhood name to filter by (e.g., 'Mission', 'Upper Market')"
            )
            RETURNS DOUBLE
            LANGUAGE SQL
            DETERMINISTIC
            COMMENT 'Calculates the average listing price for a specific neighborhood in San Francisco. Returns the average price as a numeric value. Price strings are cleaned and converted to numeric values before averaging.'
            RETURN 
            SELECT AVG(CAST(REGEXP_REPLACE(price, '[^0-9.]', '') AS DOUBLE))
            FROM sf_airbnb_listings
            WHERE neighbourhood_cleansed = neighborhood_name
            AND price IS NOT NULL
            AND REGEXP_REPLACE(price, '[^0-9.]', '') != '';
    """
    spark.sql(catalog_query)
    spark.sql(schema_query)
    spark.sql(tool1).collect()
    print(f"Created function(s): avg_neigh_price")
    return None

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP FUNCTION IF EXISTS avg_neigh_price;
# MAGIC DROP FUNCTION IF EXISTS count_properties_by_room_type;