import pandas as pd
import folium
from folium.plugins import HeatMap
from pyproj import Transformer

# Load the dataset
input_file_path = "collision_data_2017.csv"  # Output file path

# Read the dataset
df = pd.read_csv(input_file_path)
def transform_coordinates(points_list):
    """
    Transform coordinates from NY State Plane (NAD83) to WGS84 (lat/long)
    Input format: POINT (x y)
    Output: (latitude, longitude)
    """
    # Create transformer from NY State Plane (EPSG:2263) to WGS84
    transformer = Transformer.from_crs(
        "EPSG:2908",  # NY City Plane
        "EPSG:4326",  # WGS84 (lat/long)
        always_xy=True
    )

    results = []
    for point in points_list:
        # Extract x, y from POINT string
        # coords = point.replace("POINT (", "").replace(")", "").split()
        x, y = float(point[0]), float(point[1])

        # Transform coordinates
        lon, lat = transformer.transform(x, y)
        results.append((lat, lon))

    return results


def create_traffic_heatmap(data_df):
    """
    Create a heatmap of traffic volume in New York City
    """


    data_df.dropna(subset=['Latitude', 'Longitude'])
    grouped = df.groupby(['Latitude', 'Longitude']).size().reset_index(name='Counts')
    # Calculate IQR to detect outliers
    q1 = grouped['Counts'].quantile(0.25)
    q3 = grouped['Counts'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    filtered_grouped = grouped[(grouped['Counts'] >= lower_bound) & (grouped['Counts'] <= upper_bound)]

    # Combine into a single array
    heatmap_data = filtered_grouped[['Latitude', 'Longitude', 'Counts']].values.tolist()


    # Create base map centered on New York City
    meanLong = sum([point[0] for point in heatmap_data]) / len(heatmap_data)
    meanLat = sum([point[1] for point in heatmap_data]) / len(heatmap_data)
    nyc_map = folium.Map(
        location=[meanLong, meanLat],
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # Add heatmap layer
    heatmap = HeatMap(
        heatmap_data,
        min_opacity=0.41,
        radius=25,
        blur=25,
        max_zoom=1,
    )
    heatmap.add_to(nyc_map)

    return nyc_map


def main():
    # Read the data
    try:
        df = pd.read_csv('collision_data_2017.csv')
    except FileNotFoundError:
        print("The dataset.csv file was not found.")
        return

    # Create and save the map
    try:
        traffic_map = create_traffic_heatmap(df)
        traffic_map.save("nyc_collision_heatmap.html")
        print("Map saved successfully as nyc_collision_heatmap.html")
    except Exception as e:
        print(f"An error occurred while creating the map: {e}")


if __name__ == "__main__":
    main()