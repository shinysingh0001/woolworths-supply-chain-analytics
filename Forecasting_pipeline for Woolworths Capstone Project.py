import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.metrics import mean_absolute_error

# Load the Excel dataset (make sure it's in the same folder as this script)
df = pd.read_excel("Woolworths_Realistic_Simulated_Dataset.xlsx")

# Forecasting: 3-day rolling average
df['Forecast'] = df.groupby('Product Category')['Units Sold'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Forecast accuracy
valid = df.dropna(subset=['Forecast'])
mae = mean_absolute_error(valid['Units Sold'], valid['Forecast'])
accuracy = 100 - (mae / valid['Units Sold'].mean()) * 100
print(f"Forecast Accuracy: {accuracy:.2f}%")

# Simulated sentiment analysis
def simulate_sentiment(row):
    if row['Delivery Status'] == 'Delayed':
        return TextBlob("Delivery was late and frustrating").sentiment.polarity
    else:
        return TextBlob("Delivery was on time and smooth").sentiment.polarity

df['Sentiment Score'] = df.apply(simulate_sentiment, axis=1)

# Export CSV
df[['Date', 'Store Location', 'Product Category', 'Units Sold', 'Forecast', 'Sentiment Score']]\
  .to_csv("Forecast_Report_Export.csv", index=False)

# Visualization: Forecast vs Actual
df[['Units Sold', 'Forecast']].head(30).plot(figsize=(10, 5), title="Forecast vs Actual Units Sold")
plt.xlabel("Index")
plt.ylabel("Units")
plt.tight_layout()
plt.savefig("forecast_vs_actual.png")
plt.close()

# Visualization: Average Delivery Delay by Supplier
df.groupby('Supplier')['Delivery Delay (Days)'].mean().sort_values().plot(
    kind='barh', color='coral', title="Avg Delivery Delay by Supplier"
)
plt.xlabel("Delay (Days)")
plt.tight_layout()
plt.savefig("avg_delay_by_supplier.png")
plt.close()

# Visualization: Fulfillment Rate by Store Location
df.groupby('Store Location')['Fulfillment Rate (%)'].mean().plot(
    kind='pie', autopct='%1.1f%%', title="Fulfillment Rate by Store", figsize=(6,6)
)
plt.ylabel("")
plt.tight_layout()
plt.savefig("fulfillment_by_store.png")
plt.close()
