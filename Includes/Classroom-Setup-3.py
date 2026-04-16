# Databricks notebook source
# MAGIC %run ./Classroom-Setup-Common

# COMMAND ----------

# Install a coherent 0.3.x stack (compatible with unitycatalog-langchain and databricks-langchain)
%pip install -U \
  "langchain<0.4,>=0.3.27" \
  "langchain-core<0.4,>=0.3.79" \
  "langchain-community<0.4,>=0.2" \
  "langchain-text-splitters<1.0,>=0.3.9" \
  "langchain-openai<0.3,>=0.2.0" \
  "databricks-langchain==0.8.2" \
  "pydantic>=2.0.0,<3.0.0"

# Restart the Python VM so the environment picks up the new packages
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
    tool1_name = "avg_fare_by_zip"
    tool2_name = "pickup_zip_code"
    tool1 = f"""
            -- Create Tool 1 that gets the average fare by pickup zip code
            CREATE OR REPLACE FUNCTION {tool1_name}(
            pickup_zip_code INT COMMENT "The pickup ZIP code to filter by"
            )
            RETURNS DOUBLE
            LANGUAGE SQL
            DETERMINISTIC
            COMMENT 'Calculates the average fare amount for trips from a specific pickup ZIP code. Returns the average fare as a numeric value.'
            RETURN 
            SELECT AVG(fare_amount)
            FROM samples.nyctaxi.trips
            WHERE pickup_zip = pickup_zip_code
            AND fare_amount IS NOT NULL;
    """

    tool2 = f"""
        -- Create Tool 2 that counts the number of long distance trips
        CREATE OR REPLACE FUNCTION cnt_lng_dist_trip(
        {tool2_name} INT COMMENT "The pickup ZIP code to filter by",
        min_distance DOUBLE COMMENT "The minimum trip distance in miles"
        )
        RETURNS BIGINT
        LANGUAGE SQL
        DETERMINISTIC
        COMMENT 'Counts the number of trips longer than the specified distance from a given pickup ZIP code. Returns the count as an integer.'
        RETURN
        SELECT COUNT(*)
        FROM samples.nyctaxi.trips
        WHERE pickup_zip = pickup_zip_code
            AND trip_distance > min_distance;
    """
    spark.sql(catalog_query)
    spark.sql(schema_query)
    spark.sql(tool1).collect()
    spark.sql(tool2).collect()
    print(f"Created function(s): {tool1_name} and {tool2_name}")
    return None

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP FUNCTION IF EXISTS get_average_price_by_neighborhood;
# MAGIC DROP FUNCTION IF EXISTS count_properties_by_room_type;