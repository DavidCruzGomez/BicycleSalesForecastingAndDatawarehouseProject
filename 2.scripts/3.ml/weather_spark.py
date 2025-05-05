import sys

import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date


def create_spark_session():
# Sets up a Spark session for the weather analysis.

    try:
        spark_session = SparkSession.builder.appName("NOAA_weather_sales").getOrCreate()
        logging.info(f"Spark Session created successfully. Spark Version: {spark_session.version}")
        return spark_session

    except ImportError as imp_err:
        logging.error(
            "❌ Error: Could not import the Spark module.",
            exc_info=True)
        sys.exit(1)

    except Exception as err:
        logging.error("❌ Unexpected error creating the Spark session:", exc_info=True)
        sys.exit(1)


def load_weather_data(spark, path_2018, path_2019):
# Reads weather data for 2018 and 2019, performs initial date filtering.

    try:
        logging.info("Reading weather data for 2018...")
        weather_data_2018 = spark.read.csv(path_2018, inferSchema=True, header=False) \
            .selectExpr("_c0 as station_id",
                        "_c1 as raw_date",
                        "_c2 as variable",
                        "_c3 as value") \
            .withColumn("date", to_date(col("raw_date").cast("string"), "yyyyMMdd")) \
            .filter(col("date") >= "2018-01-11") \
            .drop("raw_date")

        logging.info("Reading weather data for 2019...")
        weather_data_2019 = spark.read.csv(path_2019, inferSchema=True, header=False) \
            .selectExpr("_c0 as station_id",
                        "_c1 as raw_date",
                        "_c2 as variable",
                        "_c3 as value") \
            .withColumn("date", to_date(col("raw_date").cast("string"), "yyyyMMdd")) \
            .filter(col("date") <= "2019-07-01") \
            .drop("raw_date")

        logging.info("Merging 2018 and 2019 weather data.")
        return weather_data_2018.union(weather_data_2019)

    except FileNotFoundError as fnf_error:
        logging.error(fnf_error, exc_info=True)
        raise fnf_error

    except Exception as err:
        logging.error("❌ An unexpected error occurred while loading weather data.", exc_info=True)
        raise err


def load_stations(spark, stations_path):
# Loads station metadata from a text file and filters stations to only include
# those from a given list of countries.

    try:
        logging.info(f"Loading station metadata from: {stations_path}")

        stations_raw = spark.read.text(stations_path)

        stations = stations_raw.selectExpr(
            "substring(value, 1, 11) as station_id",
            "substring(value, 13, 8) as latitude",
            "substring(value, 22, 9) as longitude",
            "substring(value, 32, 3) as elevation",
            "substring(value, 39, 2) as country_code",
            "substring(value, 42, 3) as state",
            "substring(value, 45, 30) as station_name"
        )

        countries = ['US', 'CA', 'AU', 'DE', 'FR', 'GB', 'IN', 'AE']
        filtered = stations.filter(col("country_code").isin(countries))

        logging.info(f"✅ Successfully loaded and filtered stations (total: {filtered.count()})")
        return filtered

    except FileNotFoundError as fnf_error:
        logging.error(f"❌ Station file not found: {fnf_error.filename}", exc_info=True)
        raise fnf_error
    except Exception as err:
        logging.error("❌ Error loading or processing station metadata.", exc_info=True)
        raise err


def join_weather_with_stations(weather_df, stations_df):
# Inner join between weather and station DataFrames on 'station_id'.

    try:
        logging.info("Validating columns before join...")

        if "station_id" not in weather_df.columns:
            raise ValueError("Column 'station_id' not found in weather DataFrame.")

        if "station_id" not in stations_df.columns:
            raise ValueError("Column 'station_id' not found in stations DataFrame.")

        logging.info("🔗 Joining weather data with station metadata...")
        joined_df = weather_df.join(stations_df, on="station_id", how="inner")

        row_count = joined_df.count()
        logging.info(f"✅ Join successful. Resulting rows: {row_count}")
        return joined_df

    except ValueError as val_err:
        logging.error(f"❌ Validation error: {val_err}", exc_info=True)
        raise val_err

    except Exception as err:
        logging.error("❌ Unexpected error during join operation.", exc_info=True)
        raise err


def save_to_csv(df, output_path):
# Saves a Spark DataFrame to a CSV file in a specified output directory.

    try:
        df.write.csv(output_path, header=True, mode="overwrite")
        logging.info(f"💾 Data successfully saved to: {output_path}")

    except PermissionError as per_err:
        logging.error("❌ Permission denied when writing to the output path.", exc_info=True)
        raise per_err

    except Exception as err:
        logging.error("❌ Unexpected error. Failed to save CSV file.", exc_info=True)
        raise err


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    spark = create_spark_session()
    resources_path = "../resources"
    output_path = "../outputs/filtered-weather"

    try:
        logging.info("Loading weather data...")
        weather = load_weather_data(
            spark,
            f"{resources_path}/2018.csv",
            f"{resources_path}/2019.csv"
        )
        weather.show(5)
        weather.orderBy(col("date").desc()).show(5)
        logging.info("✅ Weather data loaded.")

        stations = load_stations(spark, f"{resources_path}/ghcnd-stations.txt")

        filtered_weather = join_weather_with_stations(weather, stations)
        filtered_weather.show(5)

        logging.info("Saving final dataset...")
        save_to_csv(filtered_weather, output_path)

        logging.info("✅ Process completed successfully.")

    except Exception as err:
        logging.error("❌ An error occurred during the execution of the pipeline.", exc_info=True)

if __name__ == "__main__":
    main()
