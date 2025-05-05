# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the CSV file
pathfile = "../../0.datasets/3.ml/resources/df_cleaned.csv"
df_original = pd.read_csv(pathfile)

# Copy the DataFrame
df = df_original.copy()

# DataFrame information
print(df.info())

print(df["variable"].unique())

print(df["country_code"].unique())

weather_vars = ['TMAX', 'TMIN', 'PRCP']
df = df[df["variable"].isin(weather_vars)]

df_pivot = df.pivot_table(index=["station_id", "date"],
                          columns="variable",
                          values="value").reset_index()

print(df_pivot.info())

print(df_pivot.isna().sum())
# As the dataset has too many null values, I've opted for a smaller dataset containing all 3 records instead of imputing the mean, as TMIN and TMAX have around 60% nulls.
df_clean = df_pivot.dropna()

print(df_clean.info())
print(df_clean.head(20))

# NOAA weather data is stored in tenths of degrees Celsius and tenths of millimeters to avoid using decimal numbers in text files.
df_clean = df_clean.copy()
df_clean.loc[:, 'TMAX_C'] = df_clean['TMAX'] / 10
df_clean.loc[:, 'TMIN_C'] = df_clean['TMIN'] / 10
df_clean.loc[:, 'PRCP_MM'] = df_clean['PRCP'] / 10
df_clean = df_clean.drop(columns=['TMAX', 'TMIN', 'PRCP'])
print(df_clean)

# Create the histogram of TMAX
plt.figure(figsize=(8, 5))
plt.hist(df_clean['TMAX_C'], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of TMAX (°C)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Create the histogram of TMIN
plt.figure(figsize=(8, 5))
plt.hist(df_clean['TMIN_C'], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of TMIN (°C)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Create the histogram of PRCP
plt.figure(figsize=(8, 5))
plt.hist(df_clean['PRCP_MM'], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of Precipitations (mm)")
plt.xlabel("Precipitations (mm)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df_clean[['TMAX_C', 'TMIN_C', 'PRCP_MM']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# TMAX and TMIN are highly correlated, while PRCP has a very low correlation.

# Scatter plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_clean, x='TMIN_C', y='TMAX_C', alpha=0.5)
plt.title("TMIN vs TMAX")
plt.xlabel("TMIN")
plt.ylabel("TMAX")
plt.show()

# The scatter plot confirms a strong relationship between TMIN and TMAX.

# Save the cleaned DataFrame to a new CSV file
output_filepath = "../resources/df_weather_cleaned.csv"
df_clean.to_csv(output_filepath, index=False)

print(f"The cleaned DataFrame has been successfully saved to: {output_filepath}")
