import pandas as pd
import folium
from folium.plugins import HeatMap
from shapely import wkt
from pyproj import Transformer
import numpy as np

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


def remove_volume_outliers(heatmap_data, multiplier=1.5):
    """
    Remove outliers from heatmap data using the IQR method.

    Args:
        heatmap_data: List of [longitude, latitude, volume] lists
        multiplier: IQR multiplier for outlier detection (default 1.5)

    Returns:
        List of [longitude, latitude, volume] lists with outliers removed
    """
    # Extract volumes for outlier detection
    volumes = [point[2] for point in heatmap_data]

    # Calculate Q1, Q3, and IQR
    q1 = np.percentile(volumes, 25)
    q3 = np.percentile(volumes, 75)
    iqr = q3 - q1

    # Define outlier boundaries
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    # Filter out outliers
    filtered_data = [
        point for point in heatmap_data
        if lower_bound <= point[2] <= upper_bound
    ]

    print(f"Original data points: {len(heatmap_data)}")
    print(f"Points after outlier removal: {len(filtered_data)}")
    print(f"Removed {len(heatmap_data) - len(filtered_data)} outliers")
    print(f"Volume range: {lower_bound:.2f} to {upper_bound:.2f}")

    return filtered_data


def create_traffic_heatmap(data_df):
    """
    Create a heatmap of traffic volume in New York City
    """

    # Convert WKT geometry to coordinates
    # def extract_coordinates(wkt_string):
    #     geometry = wkt.loads(wkt_string)
    #     if geometry.is_valid:
    #         return [geometry.centroid.x, geometry.centroid.y]
    #     else:
    #         return None
    # Validate and process the data
    # data_df['coordinates'] = data_df['WktGeom'].apply(lambda x: extract_coordinates(x) if pd.notnull(x) else None)

    # Remove rows with invalid data
    valid_data = data_df.dropna(subset=['WktGeom', 'Vol'])
    valid_data = valid_data[valid_data['Vol'] > 0]  # Ensure volume is positive

    # Extract coordinates and volumes for heatmap
    locations = [[float(x.split(" ")[0]), float(x.split(" ")[1])] for x in valid_data['WktGeom'].tolist()]
    volumes = valid_data['Vol'].tolist()

    # Create the heatmap data
    heatmap_data = [[loc[0], loc[1], vol] for loc, vol in zip(locations, volumes)]
    heatmap_data = remove_volume_outliers(heatmap_data)
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
        df = pd.read_csv('dataset4.csv')
    except FileNotFoundError:
        print("The dataset.csv file was not found.")
        return

    # Create and save the map
    try:
        traffic_map = create_traffic_heatmap(df)
        traffic_map.save("nyc_traffic_heatmap.html")
        print("Map saved successfully as nyc_traffic_heatmap.html")
    except Exception as e:
        print(f"An error occurred while creating the map: {e}")


if __name__ == "__main__":
    main()
