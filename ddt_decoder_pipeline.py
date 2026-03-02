# Import necessary libraries
from decode_func import decode_ddt
import json
import pandas as pd
import os

from pyspark.sql.functions import udf, col, to_timestamp, length
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType

from helpers.alert_string_lengths import alrt_len

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from databricks.connect import DatabricksSession

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# Load configuration from config.json
with open('helpers/config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize Databricks session for Databricks Connect
# spark = DatabricksSession.builder.getOrCreate()
# sc = spark.sparkContext

def main(go_fast=False, sampleData=True):

    # #========================================================
    # # Step 1: Load and Prepare Vehicle Data 
    # #========================================================
    # print("\n=== Step 1: Load and Prepare Vehicle Data ===")
    
    # # Load data 
    # df = load_data()

    # #  Add "MODEL_YEAR", "BRANDNAME", "FAMILY"
    # df = join_vehicle_data(df)

    # #=========================================================   
    # # Step 2: Analyze Available Families ===    
    # #=========================================================
    # print("\n=== Step 2: Analyze Available Families and Filter by String Length ===")

    # families_with_conversions, available_families, family_year_combinations, df_filtered = availFamilies(df, go_fast=go_fast)

    # print("Families with conversions:", families_with_conversions)
    # print("Available families:", available_families)
    # print("Family-year combinations with conversions:", family_year_combinations)

    # # Use the filtered DataFrame from this point forward
    # df = df_filtered


    # #=========================================================
    # # Step 3: Create Sample Dataset
    # #=========================================================
    # print("\n=== Step 3: Create Sample Dataset ===")
    
    # if sampleData:
    #     # Create a sample dataset with 50 rows for each family
    #     df = create_sample_dataset(df, families_with_conversions, go_fast=go_fast)

    # # Save the DataFrame to the specified table
    # df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("gadp_scratch.rich.scratch")
    # print(family_year_combinations)

    # return("stopped")

    family_year_combinations = [('DT', '2024'), ('DT', '2025'), ('JL', '2024'), ('KM', '2024'), ('LB', '2024'), ('WL', '2024'), ('WS', '2025')]
    df = spark.read.table("gadp_scratch.rich.scratch")
  
    #=========================================================
    # Step 4: Apply DDT Decoding
    #=========================================================
    print("\n=== Step 4: Apply DDT Decoding ===")
    ## Let's loop through and decode each family seperately 

    # Initialize an empty list to store decoded dataframes
    decoded_dataframes = []     

    for fam_yr in family_year_combinations:
        print(f"\nDecoding family: {fam_yr[1]}-{fam_yr[0]}")
        
        # Load conversion tables
        conv_tbl_pd_L, conv_tbl_pd_T = loadConvTables(fam_yr)
        
        if conv_tbl_pd_L is None or conv_tbl_pd_T is None:
            print(f"Skipping {fam_yr} - conversion tables not found")
            continue

        # Filter df to only include rows with the current family and model year
        df_filtered = df.filter((col("MODEL_CODE") == fam_yr[0]) & (col("MODEL_YEAR") == fam_yr[1]))

        # Check if we have any data for this family-year combination
        if df_filtered.count() == 0:
            print(f"No data found for {fam_yr[0]}-{fam_yr[1]}, skipping...")
            continue

        display(df_filtered)

        # Find available DDT alerts
        ddt_alerts = findAlerts(df_filtered)
        print(f"DDT alerts found: {ddt_alerts}")

        # Create DDT file mapping
        DDT_files = {}
        for ddt_alert in ddt_alerts:
            DDT_files[ddt_alert] = f'DDT_BCM_{ddt_alert}_{fam_yr[0]}_{fam_yr[1]}.json'
        print(f'DDT_file: {DDT_files}')


        # Create UDFs for each alert with proper closures
        df_decoded = df_filtered
        for alert in ddt_alerts:
            # Create a decoder function with all logic self-contained
            def create_decoder(alert_type, family, year, conv_L, conv_T, ddt_file_map, config_paths):
                def decode_wrapper(ddt_string):
                    if not ddt_string:
                        return json.dumps({"error": "Empty DDT string"})
                    
                    try:
                        # Construct the file path for the DDT structure
                        ddt_filename = ddt_file_map.get(alert_type)
                        if not ddt_filename:
                            return json.dumps({"error": f"DDT file mapping not found for {alert_type}"})
                            
                        file_path = os.path.join(config_paths['conversions'], family, year, ddt_filename)
                        
                        # Check if file exists
                        if not os.path.exists(file_path):
                            return json.dumps({"error": f"Structure file not found: {file_path}"})
                            
                        # Load the JSON structure
                        with open(file_path, 'r') as file:
                            json_structure = json.load(file)
                            
                        if not json_structure:
                            return json.dumps({"error": f"Empty structure for {alert_type}"})
                        
                        # Import decode_ddt inside the function to ensure it's available
                        from decode_func import decode_ddt
                        
                        # Decode the DDT string
                        result = decode_ddt(ddt_string, json_structure, conv_T, conv_L)
                        
                        # Add vehicle info to the result
                        result["vehicle_family"] = family
                        result["model_year"] = year
                        result["ddt_type"] = alert_type
                        
                        # Convert to JSON string for return
                        return json.dumps(result)
                        
                    except Exception as e:
                        return json.dumps({
                            "error": str(e),
                            "error_type": str(e.__class__.__name__),
                            "ddt_type": alert_type,
                            "family": family,
                            "year": year
                        })
                return decode_wrapper
            
            # Create the UDF with captured variables
            decode_udf = udf(
                create_decoder(alert, fam_yr[0], fam_yr[1], 
                             conv_tbl_pd_L, conv_tbl_pd_T, DDT_files, config['paths']), 
                StringType()
            )
            
            # Apply the UDF
            df_decoded = df_decoded.withColumn(
                f"decoded_BCM_{alert}",
                decode_udf(col(f"DIAG_BCM_{alert}"))
            )
        #=================================================
        # Flatten the decoded column
        #=================================================
        df_decoded_flattened = json_flatening(df_decoded, ddt_alerts, fam_yr)
        
        # Add this family's decoded dataframe to our list
        decoded_dataframes.append(df_decoded_flattened)
        
        print(f"Completed decoding for {fam_yr[0]}-{fam_yr[1]}")
        display(df_decoded_flattened)

    #=================================================
    # Union all decoded dataframes together
    #=================================================
    print(f"\n=== Step 5: Union All Decoded Dataframes ===")

    if decoded_dataframes:
        # Start with the first dataframe
        final_decoded_df = decoded_dataframes[0]
        
        # Union with all subsequent dataframes
        for i, df_to_union in enumerate(decoded_dataframes[1:], 1):
            print(f"Unioning dataframe {i+1} of {len(decoded_dataframes)}")
            try:
                # Use unionByName to handle potential schema differences
                final_decoded_df = final_decoded_df.unionByName(df_to_union, allowMissingColumns=True)
            except Exception as e:
                print(f"Error during union operation: {e}")
                print("Attempting regular union...")
                try:
                    final_decoded_df = final_decoded_df.union(df_to_union)
                except Exception as e2:
                    print(f"Regular union also failed: {e2}")
                    print("Skipping this dataframe...")
                    continue
        
        print(f"Final combined dataframe has {final_decoded_df.count()} rows")
        print("Schema of final dataframe:")
        final_decoded_df.printSchema()
        
        # Display summary by family
        print("Summary by family:")
        final_decoded_df.groupBy("MODEL_CODE", "MODEL_YEAR").count().orderBy("MODEL_CODE", "MODEL_YEAR").show()
        
        # Display the final result
        display(final_decoded_df)
        
        # Save the final result
        final_decoded_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("ds.12v_battery.area51_decode_pipeline")
        
    else:
        print("No dataframes were successfully decoded!")
        final_decoded_df = None

    
    return print('Pipeline completed successfully')



#==========================
#==========================

def load_data():
    """Load data from the specified table"""
    try:
        df = spark.read.table(config["input_data"]["ada_datastream"])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def join_vehicle_data(df):
    """Join vehicle data with diagnostic data"""
    try:
        df_vehicle = spark.read.table(config["input_data"]["vehicle_data"])\
            .select("VIN", "MODEL_YEAR", F.expr("substring(MODEL_CODE, 1, 2)").alias("MODEL_CODE"), "MODEL_NAME")

        # Join vehicle data with diagnostic data
        df_joined = df.join(df_vehicle, df["vin"] == df_vehicle["VIN"], "inner")\
            .select(df["vin"], df_vehicle["MODEL_YEAR"], df_vehicle["MODEL_CODE"], df_vehicle["MODEL_NAME"], "app_timestamp", 
                    col("ibs_tcsm_22a001").alias('DIAG_BCM_A001'),\
                    col("ibs_tcsm_22a001").alias('DIAG_BCM_A007')
                    # col("bcm_22a0b5").alias('DIAG_BCM_A0B5'),\
                    # col("bcm_22a0b6").alias('DIAG_BCM_A0B6'),\
                    # col("bcm_22a0b7").alias('DIAG_BCM_A0B7'),\
                    # col("bcm_220129").alias('DIAG_BCM_0129')
                )
        return df_joined
    except Exception as e:
        print(f"Error joining vehicle data: {e}")
        return None
    
def findAlerts(df):
    """Find alerts in the diagnostic data
    Create a list of DDT alerts in this dataset"""
    ddt_alerts = [col.split('_', 2)[2] for col in df.columns if col.startswith('DIAG_') and len(col.split('_')) > 2]
    return ddt_alerts

def analyze_families(df):
    """Analyze the dataset to identify all available families"""
    try:
        # Get unique families and their counts from the joined data
        # Analyze the dataset to identify all available families
        print("Analyzing dataset to identify all available vehicle families...")

        # Get unique families and their counts from the joined data
        families_summary = df.groupBy("MODEL_CODE", "MODEL_YEAR").count().orderBy("MODEL_CODE", "MODEL_YEAR")
        print("Families in dataset:")
        families_summary.show()
       

        # Also check what conversion files are available
        base_conv_path = config["paths"]["conversions"]
        available_families = []
        if os.path.exists(base_conv_path):
            for family in os.listdir(base_conv_path):
                if os.path.isdir(os.path.join(base_conv_path, family)):
                    available_families.append(family)
                    
        print(f"Available conversion files for families: {sorted(available_families)}")

        # Check which families have both data and conversion files
        data_families = [row['MODEL_CODE'] for row in families_summary.select('MODEL_CODE').distinct().collect()]
        families_with_conversions = list(set(data_families) & set(available_families))
        print(f"Families with both data and conversion files: {sorted(families_with_conversions)}")

    except Exception as e:
        print(f"Error analyzing families: {e}")
        return None


def load_ddt_structure(family, year, ddt_type):
    """Load DDT structure file for a specific family and year"""
    dir_path = f"/Workspace/Users/richard.sajdak@stellantis.com/dsci_12V_battery_ddt_decode/conversions/{family}/{year}/"
    ddt_file = f'DDT_BCM_{ddt_type}_{family}_{year}.json'
    
    try:
        file_path = os.path.join(dir_path, ddt_file)
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading DDT structure for {family}/{year}/{ddt_type}: {e}")
        return None



def availFamilies(df, go_fast=False):
    """
    Analyze the dataset to identify all available families and their model years.
    Filter for diagnostic messages with correct string lengths.
    
    Args:
        df: DataFrame containing the diagnostic data
        go_fast: Boolean flag to skip detailed output for faster execution
        
    Returns:
        tuple: (families_with_conversions, available_families, family_year_combinations)
            - families_with_conversions: List of families that have conversion files
            - available_families: List of all families with conversion files available
            - family_year_combinations: List of tuples (family, year) for families with conversions
    """
    # Import the alert string lengths
    from helpers.alert_string_lengths import alrt_len
    
    # Analyze the dataset to identify all available families
    print("Analyzing dataset to identify all available vehicle families...")

    # Get unique families and their counts from the joined data
    families_summary = df.groupBy("MODEL_CODE", "MODEL_YEAR").count().orderBy("MODEL_CODE", "MODEL_YEAR")
    
    if go_fast == False:
        print("Families in dataset:")
        families_summary.show()

    # Also check what conversion files are available
    base_conv_path = config["paths"]["conversions"]
    print(f"\nChecking conversion files in: {base_conv_path}")
    available_families = []
    available_family_years = []  # New: store family-year combinations
    
    if os.path.exists(base_conv_path):
        for family in os.listdir(base_conv_path):
            family_path = os.path.join(base_conv_path, family)
            if os.path.isdir(family_path):
                available_families.append(family)
                
                # Check for available years within each family directory
                try:
                    for year in os.listdir(family_path):
                        year_path = os.path.join(family_path, year)
                        if os.path.isdir(year_path):
                            # Verify that required conversion files exist
                            conv_L_file = f'conv_L_BCM_{family}_{year}.csv'
                            conv_T_file = f'conv_T_BCM_{family}_{year}.csv'
                            
                            if (os.path.exists(os.path.join(year_path, conv_L_file)) and 
                                os.path.exists(os.path.join(year_path, conv_T_file))):
                                available_family_years.append((family, year))
                except Exception as e:
                    print(f"Error checking years for family {family}: {e}")
                    
    print(f"   Available conversion files for families: {sorted(available_families)}")
    print(f"   Available family-year combinations: {sorted(available_family_years)}")
    print("\n   Checking for family-year combinations that has both data and conversion files...")

    # Check which families have both data and conversion files
    data_families = [row['MODEL_CODE'] for row in families_summary.select('MODEL_CODE').distinct().collect()]
    families_with_conversions = list(set(data_families) & set(available_families))
    
    # Get family-year combinations that exist in both data and conversion files
    data_family_years = [(row['MODEL_CODE'], str(row['MODEL_YEAR'])) for row in families_summary.collect()]
    family_year_combinations_with_conversions = list(set(data_family_years) & set(available_family_years))
    
    # NEW: Filter DataFrame for diagnostic messages with correct string lengths
    print("\n   Filtering for diagnostic messages with correct string lengths...")
    
    # Start with a base condition that's always false, then OR with valid conditions
    length_filter_condition = F.lit(False)
    
    for fam_var in families_with_conversions:
        if fam_var in alrt_len:
            # Now iterate through model years within each family
            for model_year in alrt_len[fam_var]:
                # Convert model year to string for consistency with DataFrame
                model_year_str = str(model_year)
                
                # Iterate through diagnostic alerts for this family-year combination
                for diag_alrt in alrt_len[fam_var][model_year]:
                    # Check if the diagnostic column exists in the DataFrame
                    diag_col_name = f"DIAG_BCM_{diag_alrt}"
                    if diag_col_name in df.columns:
                        # Build the condition for this family, model year, and alert combination
                        condition = (
                            (col("MODEL_CODE") == fam_var) & 
                            (col("MODEL_YEAR") == model_year_str) &
                            (length(col(diag_col_name)) == alrt_len[fam_var][model_year][diag_alrt])
                        )
                        # OR this condition with the existing filter
                        length_filter_condition = length_filter_condition | condition
                        
                        if not go_fast:
                            print(f"   Added filter: {fam_var} - {model_year_str} - {diag_alrt} - expected length: {alrt_len[fam_var][model_year][diag_alrt]}")
    
    # Apply the length filter to the DataFrame
    df_filtered = df.filter(length_filter_condition)
    
    # Check the results of filtering
    if not go_fast:
        print("\n   Counts after string length filtering:")
        filtered_counts = df_filtered.groupBy("MODEL_CODE", "MODEL_YEAR").count().orderBy("MODEL_CODE", "MODEL_YEAR")
        filtered_counts.show()
        
        # Show original vs filtered counts
        original_total = df.count()
        filtered_total = df_filtered.count()
        print(f"   Original total rows: {original_total}")
        print(f"   Filtered total rows: {filtered_total}")
        print(f"   Rows removed: {original_total - filtered_total}")
    
    if go_fast == False:
        print(f"Families with both data and conversion files: {sorted(families_with_conversions)}")
        print(f"Family-year combinations with both data and conversion files: {sorted(family_year_combinations_with_conversions)}")
                
    return (families_with_conversions, sorted(available_families), sorted(family_year_combinations_with_conversions), df_filtered)


def create_sample_dataset(df, families_with_conversions, go_fast=False):
    """
    Create a sample dataset with 50 rows for each family in families_with_conversions.
    
    Args:
        df: DataFrame containing the diagnostic data
        families_with_conversions: List of families to sample
        
    Returns:
        Sampled DataFrame with 50 rows per family
    """
    print(f"Creating sample with 50 rows for each family: {families_with_conversions}")
    
    df = df.filter(col("DIAG_BCM_A001").isNotNull())

    # Create a window specification to number rows within each family
    window = Window.partitionBy("MODEL_CODE").orderBy("app_timestamp")

    # Add row numbers within each family and filter to get 50 rows per family
    df_sample = df.filter(col("MODEL_CODE").isin(families_with_conversions)) \
        .withColumn("row_num", F.row_number().over(window)) \
        .filter(col("row_num") <= 50) \
        .drop("row_num")
    
    if go_fast == False:
        # Verify the sampling worked correctly
        sample_counts = df_sample.groupBy("MODEL_CODE").count().orderBy("MODEL_CODE")
        print("Sample counts by family:")
        sample_counts.show()

    return df_sample

def loadConvTables(fam_yr):
    """Load conversion tables for a specific family and year"""

    conv_L = f'conv_L_BCM_{fam_yr[0]}_{fam_yr[1]}.csv'
    conv_T = f'conv_T_BCM_{fam_yr[0]}_{fam_yr[1]}.csv'

    try:
        conv_tbl_T = spark.read.csv(
            f"file://{config['paths']['conversions']}{fam_yr[0]}/{fam_yr[1]}/{conv_T}",
            header=True,
            inferSchema=True
        )
        # Convert to pandas for easier filtering in the decode function
        conv_tbl_pd_T = conv_tbl_T.toPandas()
        # Ensure ID is string type for consistent comparisons
        # conv_tbl_pd_T['ID'] = conv_tbl_pd_T['ID'].astype(str)

        #===============

        # Load linear conversion table
        # Read the CSV file
        conv_tbl_L = spark.read.csv(
            f"file://{config['paths']['conversions']}{fam_yr[0]}/{fam_yr[1]}/{conv_L}",
            header=True,
            inferSchema=True
        )
        # Convert to pandas for easier filtering in the decode function
        conv_tbl_pd_L = conv_tbl_L.toPandas()

        return conv_tbl_pd_L, conv_tbl_pd_T
    except Exception as e:
        print(f"Error loading conversion tables for {fam_yr[0]}/{fam_yr[1]}: {e}")
        return None, None

def decode_ddt_wrapper(ddt_string, structure_name, family=None, model_year=None, ddt_type='A001'):
    """Wrapper for the decode_ddt function that works with UDFs and uses vehicle-specific conversions.
    
    Args:
        ddt_string: String containing the hex DDT code
        structure_name: Name of the structure to use for decoding
        family: Vehicle family (optional)
        model_year: Vehicle model year (optional)
        ddt_type: Type of DDT to decode (A001 or A007)
        
    Returns:
        JSON string with decoded data
    """
    if not ddt_string:
        return json.dumps({"error": "Empty DDT string"})
    
    try:
        # Get structure - fix the path to use conversions directory
        file_path = os.path.join(f"{config['paths']['conversions']}{family}/{model_year}/", DDT_file[ddt_type])
        print(f"IN DECODE_DDT_WRAPPER. Loading structure from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return json.dumps({"error": f"Structure file not found: {file_path}"})
            
        with open(file_path, 'r') as file:
            json_structure = json.load(file)
            
        if not json_structure:
            return json.dumps({"error": f"Structure {structure_name} not found"})
        
        # Get the appropriate conversion tables
        if family and model_year:
            # Use the broadcast conversion tables
            conv_table_L = conv_tbl_pd_L_broadcast.value
            conv_table_T = conv_tbl_pd_T_broadcast.value
        else:
            # Fall back to the default broadcast tables if no family/year specified
            conv_table_L = conv_tbl_pd_L_broadcast.value
            conv_table_T = conv_tbl_pd_T_broadcast.value
        
        # Use the appropriate conversion tables
        result = decode_ddt(ddt_string, json_structure, conv_table_T, conv_table_L)
        
        # Add vehicle info to the result
        if family and model_year:
            result["vehicle_family"] = family
            result["model_year"] = model_year
        
        # Convert to JSON string for return
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": str(e.__class__.__name__)})
    
def optimized_flatten_json(df, diag_alrt, fam_yr, json_column_name, column_types=None):
    """
    Flattens a JSON column using Spark's native JSON functions.
    Casts columns to specified data types from a dictionary.
    
    Args:
        df: DataFrame containing the JSON column
        json_column_name: Name of the column containing the JSON strings
        column_types: Dictionary mapping column names to data types
                     (e.g., {'Engine_Speed': 'int', 'Vehicle_Status': 'string'})
                     Supported types: 'string', 'int', 'float', 'boolean'
        
    Returns:
        DataFrame with flattened columns cast to specified types
    """

    # Initialize column_types if not provided
    if column_types is None:
        column_types = {}
        
    # Map string type names to PySpark types
    type_mapping = {
        'string': StringType(),
        'str': StringType(),
        'int': IntegerType(),
        'integer': IntegerType(),
        'float': FloatType(),
        'double': FloatType(),
        'bool': BooleanType(),
        'boolean': BooleanType()
    }

    # Build the query string dynamically with error handling
    try:
        expected_length = alrt_len[fam_yr[0]][fam_yr[1]][diag_alrt]
        query_string = f'(length(col("DIAG_BCM_{diag_alrt}")) == {expected_length})'
    except KeyError as e:
        print(f"Warning: Missing alert length definition for family={fam_yr[0]}, year={fam_yr[1]}, alert={diag_alrt}")
        print(f"Available families: {list(alrt_len.keys())}")
        if fam_yr[0] in alrt_len:
            print(f"Available years for {fam_yr[0]}: {list(alrt_len[fam_yr[0]].keys())}")
            if fam_yr[1] in alrt_len[fam_yr[0]]:
                print(f"Available alerts for {fam_yr[0]}/{fam_yr[1]}: {list(alrt_len[fam_yr[0]][fam_yr[1]].keys())}")
        
        # Fallback: create a query that doesn't filter by length
        query_string = f'col("DIAG_BCM_{diag_alrt}").isNotNull()'
        print(f"Using fallback query: {query_string}")

    df_good_row = df.filter(eval(query_string)).limit(1).collect()

    descriptions = {}

    # Check if we got any rows and if the column exists
    if df_good_row and json_column_name in df_good_row[0].asDict():
        try:
            # Get the JSON string from the first (and only) row
            json_data = df_good_row[0][json_column_name]
            if json_data:  # Check if the JSON data is not null
                data = json.loads(json_data)
                for entry in data.get("decoded_values", []):
                    desc = entry.get("description")
                    if desc and desc != "Reserved":
                        descriptions[desc] = descriptions.get(desc, "")
        except Exception as e:
            print(f"Error parsing JSON: {e}")

    # Create a clean column name function
    def clean_column_name(name):
        import re
        base_name = re.sub(r'[^a-zA-Z0-9_]', '_', name).replace('__', '_')
        return base_name

    # Start with the original dataframe
    result_df = df

    # For each description, create a new column using expressions
    for desc, unit in sorted(descriptions.items()):
        clean_desc = clean_column_name(desc)
        
        # expression to find the matching description and extract its value
        expr_str = f"""
        TRANSFORM(
            FILTER(
                from_json({json_column_name}, 'struct<decoded_values:array<struct<description:string,decoded_value:string,units:string>>>').decoded_values,
                x -> x.description = '{desc}'
            ),
            x -> x.decoded_value
        )[0]
        """
        
        # Add the new column to the DataFrame
        result_df = result_df.withColumn(clean_desc, F.expr(expr_str))
        
        # Cast the column to the specified type if provided
        # Look for exact match or pattern match in column_types dictionary
        column_type = None
        for pattern, dtype in column_types.items():
            if pattern == clean_desc or (
                    pattern.endswith('*') and clean_desc.startswith(pattern[:-1])
            ):
                column_type = dtype
                break
                
        if column_type and column_type.lower() in type_mapping:
            # Create safe casting with null handling
            result_df = result_df.withColumn(
                clean_desc,
                F.when(col(clean_desc).isNull(), None)
                .otherwise(col(clean_desc).cast(type_mapping[column_type.lower()]))
            )

    # Add the goodDiagInput column with error handling
    try:
        result_df = result_df.withColumn(
            f"goodDiagInput_{diag_alrt}",
            F.when(eval(query_string), "x").otherwise("")
        )
    except Exception as e:
        print(f"Error adding goodDiagInput column: {e}")
        # Fallback: add a simple column
        result_df = result_df.withColumn(
            f"goodDiagInput_{diag_alrt}",
            F.lit("unknown")
        )
    
    return result_df



def json_flatening(df_decoded, ddt_alerts, fam_yr):
    # Import the alert string lengths
    from helpers.column_data_types import column_types_dict

    d1 = None
    # Start with the decoded dataframe
    d1 = df_decoded

    # Loop through all alerts and flatten each one
    for alert in ddt_alerts:
        # Get the column types for this alert, or use an empty dict if not defined
        alert_column_types = column_types_dict.get(alert, {})

        # Flatten the JSON column for this alert
        d1 = optimized_flatten_json(
            d1,
            alert,
            fam_yr,
            f"decoded_BCM_{alert}",
            column_types=alert_column_types
        )
    return d1


if __name__ == "__main__":
    # Execute main function when script is run directly
    result_df = main(go_fast=False, sampleData=True)

    print("DDT Decoder script completed successfully!")


