import plotly.express as px
import pandas as pd

# Load and prepare data
df = pd.read_csv("dataset4.csv")

df = df.groupby(["fromSt", "toSt"]).agg({
    "Vol": "mean",
    "WktGeom": "first"  # keeps the first street name encountered,
}).reset_index()
df[['lat', 'lon']] = df['WktGeom'].str.split(' ', expand=True).astype(float)
# Clean volume data - convert negatives to zero and scale for better visualization
df['Vol_cleaned'] = df['Vol'].clip(lower=0)  # Remove negative values
df['Vol_scaled'] = df['Vol_cleaned'] + 1  # Add 1 to ensure all values are positive

fig = px.scatter_mapbox(df,
    lon=df["lon"],
    lat=df["lat"],
    zoom=13,
    color=df["Vol"],       # Keep original values for color
    size=df["Vol_scaled"], # Use cleaned values for size
    size_max=30,          # Maximum marker size
    width=1200,
    height=900,
)

fig.update_layout(mapbox_style="open-street-map")
fig.write_html("weight.html")