from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import geopandas as gpd
import json
from shapely.geometry import Point, LineString
import numpy as np
import os
import geocoder
import requests
from pyproj import CRS
import math
from shapely.ops import transform
from functools import partial
import pyproj
import pandas as pd
from shapely.ops import unary_union
import networkx as nx
from shapely.ops import nearest_points

# Set environment variable to restore .shx file
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# --- Function Definitions ---

# Function to load and preprocess the shapefile
def load_shapefile(filepath):
    print(f"Loading shapefile from {filepath}...")
    try:
        gdf = gpd.read_file(filepath)
        print(f"Shapefile loaded with {len(gdf)} features")
        print(f"Shapefile columns: {gdf.columns.tolist()}")
        
        # Ensure CRS is set, default to 4326 if not present
        if gdf.crs is None:
            print("Warning: Shapefile CRS is not set. Assuming EPSG:4326.")
            gdf.crs = "EPSG:4326"
        else:
            print(f"Shapefile CRS: {gdf.crs}")

        # Convert to EPSG:4326 for web mapping
        if gdf.crs != "EPSG:4326":
            print("Converting to EPSG:4326...")
            gdf = gdf.to_crs(epsg=4326)
            print("Conversion complete")
        
        # Convert GeoDataFrame to GeoJSON
        print("Converting to GeoJSON...")
        geojson_data = json.loads(gdf.to_json())
        print("GeoJSON conversion complete")
        print(f"GeoJSON features: {len(geojson_data['features'])}")
        print(f"Final CRS: {gdf.crs}")

        # Validate geometry
        gdf['is_valid'] = gdf.geometry.is_valid
        invalid_count = len(gdf[~gdf['is_valid']])
        if invalid_count > 0:
            print(f"Warning: Found {invalid_count} invalid geometries.")
            # Optionally, try to fix invalid geometries
            # gdf.geometry = gdf.geometry.buffer(0)
            # gdf = gdf[gdf.geometry.is_valid] 
            # print(f"Removed invalid geometries, {len(gdf)} remain.")
        
        # Remove rows with empty or invalid geometries if necessary
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf[gdf.geometry.is_valid]
        
        # Ensure geometries are LineString (needed for graph building)
        gdf = gdf[gdf.geometry.geom_type == 'LineString']
        print(f"Filtered to {len(gdf)} valid LineString features.")

        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# Function to geocode address using Nominatim
def geocode_address(address):
    """Convert address to coordinates using OpenStreetMap Nominatim"""
    try:
        # Clean up the address
        address = address.strip()
        
        # If it's already a coordinate pair (e.g., "41.74094838347696, -111.82899103620227")
        if ',' in address and all(part.strip().replace('.', '').replace('-', '').isdigit() for part in address.split(',')):
            lat, lon = map(float, address.split(','))
            return lat, lon
        
        # Add "Logan, Utah" to the address if not already present
        if "logan" not in address.lower():
            address = f"{address}, Logan, Utah"
        
        # Add User-Agent header to the request
        headers = {
            'User-Agent': 'BikeRoutePlanner/1.0'
        }
        
        print(f"Attempting to geocode address: {address}")  # Debug log
        
        # Use requests instead of geocoder for better control
        response = requests.get(
            f"https://nominatim.openstreetmap.org/search",
            params={
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            },
            headers=headers
        )
        
        print(f"Geocoding response status: {response.status_code}")  # Debug log
        
        if response.status_code == 200:
            results = response.json()
            if results:
                print(f"Found coordinates: {results[0]['lat']}, {results[0]['lon']}")  # Debug log
                return float(results[0]['lat']), float(results[0]['lon'])
            else:
                print(f"No results found for address: {address}")
                return None
        else:
            print(f"Error geocoding address: {response.status_code}")
            print(f"Response content: {response.text}")  # Debug log
            return None
            
    except Exception as e:
        print(f"Error geocoding address: {e}")
        return None

def calculate_route_info(route_line, gdf):
    """Calculate detailed route information based on intersections with bike infrastructure"""
    try:
        # Create a GeoDataFrame from the route line with CRS set to EPSG:4326
        route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
        print(f"Route GeoDataFrame created with CRS: {route_gdf.crs}")
        
        # Ensure both GeoDataFrames are in the same CRS
        if route_gdf.crs != gdf.crs:
            print(f"Converting route from {route_gdf.crs} to {gdf.crs}")
            route_gdf = route_gdf.to_crs(gdf.crs)
        
        # Find intersections with bike infrastructure
        print("Calculating intersections...")
        intersections = gpd.overlay(route_gdf, gdf, how='intersection')
        print(f"Found {len(intersections)} intersections")
        
        if len(intersections) == 0:
            print("No intersections found with bike infrastructure")
            return {
                'comfort_score': 0,
                'total_length': route_line.length * 0.000621371,  # Convert meters to miles
                'bike_infrastructure_length': 0,
                'intersection_count': 0,
                'average_comfort_level': 0,
                'infrastructure_types': []
            }
        
        # Calculate lengths
        total_length_meters = route_line.length
        bike_length_meters = intersections.geometry.length.sum()
        
        # Convert to miles
        total_length_miles = total_length_meters * 0.000621371
        bike_length_miles = bike_length_meters * 0.000621371
        
        print(f"Total route length: {total_length_miles} miles")
        print(f"Bike infrastructure length: {bike_length_miles} miles")
        
        # Calculate comfort score (0-100)
        comfort_score = (bike_length_meters / total_length_meters) * 100
        
        # Get unique infrastructure types
        infrastructure_types = []
        if 'Infrastructure' in intersections.columns:
            infrastructure_types = intersections['Infrastructure'].unique().tolist()
            print(f"Infrastructure types found: {infrastructure_types}")
        
        # Calculate average comfort level
        avg_comfort = 0
        if 'ComfortLvl' in intersections.columns:
            avg_comfort = intersections['ComfortLvl'].mean()
            print(f"Average comfort level: {avg_comfort}")
        else:
            print("No ComfortLvl column found in intersections")
            print(f"Available columns: {intersections.columns.tolist()}")
        
        return {
            'comfort_score': round(comfort_score, 2),
            'total_length': round(total_length_miles, 2),
            'bike_infrastructure_length': round(bike_length_miles, 2),
            'intersection_count': len(intersections),
            'average_comfort_level': round(avg_comfort, 2),
            'infrastructure_types': infrastructure_types
        }
    except Exception as e:
        print(f"Error in calculate_route_info: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'comfort_score': 0,
            'total_length': route_line.length * 0.000621371,
            'bike_infrastructure_length': 0,
            'intersection_count': 0,
            'average_comfort_level': 0,
            'infrastructure_types': []
        }

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points using the Haversine formula"""
    R = 3959.87433  # Earth's radius in miles

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c

    return round(distance, 2)

def calculate_road_distance(start_coords, end_coords):
    """Calculate the distance along roads using OpenStreetMap routing"""
    try:
        # Use OSRM routing service
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?geometries=geojson"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 'Ok':
                # Convert distance from meters to miles
                distance_miles = data['routes'][0]['distance'] * 0.000621371
                # Get the route geometry
                route_geometry = data['routes'][0]['geometry']
                return {
                    'distance': round(distance_miles, 2),
                    'geometry': route_geometry
                }
        
        return None
    except Exception as e:
        print(f"Error calculating road distance: {str(e)}")
        return None

def find_comfortable_routes(start_point, end_point, gdf, num_routes=4):
    """Find the most comfortable routes between two points"""
    try:
        print(f"Searching for routes between {start_point.coords[0]} and {end_point.coords[0]}")
        
        # Create a line between start and end points
        direct_line = LineString([start_point, end_point])
        direct_distance = direct_line.length
        
        # Calculate buffer size based on the direct distance
        buffer_distance = max(0.05, direct_distance * 0.2)
        print(f"Initial buffer distance: {buffer_distance} degrees")

        # Create buffers around points and the direct line
        start_buffer = start_point.buffer(buffer_distance)
        end_buffer = end_point.buffer(buffer_distance)
        route_buffer = direct_line.buffer(buffer_distance)

        # Combine all buffers for the search area
        search_area = unary_union([start_buffer, end_buffer, route_buffer])
        
        # Find all roads within the search area
        potential_roads = gdf[gdf.geometry.intersects(search_area)]
        print(f"Found {len(potential_roads)} roads in initial search area")

        if len(potential_roads) < num_routes:
            # If not enough roads found, increase buffer size
            buffer_distance = buffer_distance * 2
            print(f"Increasing buffer to {buffer_distance} degrees")
            search_area = unary_union([
                start_point.buffer(buffer_distance),
                end_point.buffer(buffer_distance),
                direct_line.buffer(buffer_distance)
            ])
            potential_roads = gdf[gdf.geometry.intersects(search_area)]
            print(f"Found {len(potential_roads)} roads after increasing buffer")

        if len(potential_roads) == 0:
            print("No roads found in search area")
            return None

        # Calculate comfort scores for each road
        comfort_metrics = ['SpeedComfo', 'LaneComfor', 'TrafficCom', 'CrimeComfo', 'EnergyComf']
        
        # Ensure all comfort metrics are numeric
        for metric in comfort_metrics:
            if metric in potential_roads.columns:
                potential_roads[metric] = pd.to_numeric(potential_roads[metric], errors='coerce').fillna(0)
        
        # Calculate weighted comfort score
        weights = {
            'SpeedComfo': 0.2,
            'LaneComfor': 0.2,
            'TrafficCom': 0.3,
            'CrimeComfo': 0.15,
            'EnergyComf': 0.15
        }
        
        # Calculate weighted comfort score for each road
        potential_roads['comfort_score'] = 0
        for metric, weight in weights.items():
            if metric in potential_roads.columns:
                # Normalize the metric to 0-1 range
                min_val = potential_roads[metric].min()
                max_val = potential_roads[metric].max()
                if max_val > min_val:
                    normalized = (potential_roads[metric] - min_val) / (max_val - min_val)
                else:
                    normalized = potential_roads[metric]
                potential_roads['comfort_score'] += normalized * weight

        # Add a distance field to help prioritize relevant roads
        potential_roads['distance_to_points'] = potential_roads.apply(
            lambda row: min(
                row.geometry.distance(start_point),
                row.geometry.distance(end_point)
            ),
            axis=1
        )

        # Calculate road length
        potential_roads['length'] = potential_roads.geometry.length

        # Normalize scores
        max_comfort = potential_roads['comfort_score'].max()
        min_comfort = potential_roads['comfort_score'].min()
        max_dist = potential_roads['distance_to_points'].max()
        min_dist = potential_roads['distance_to_points'].min()
        max_length = potential_roads['length'].max()
        min_length = potential_roads['length'].min()

        # Normalize comfort score (higher is better)
        if max_comfort != min_comfort:
            potential_roads['normalized_comfort'] = (potential_roads['comfort_score'] - min_comfort) / (max_comfort - min_comfort)
        else:
            potential_roads['normalized_comfort'] = 1

        # Normalize distance (closer is better)
        if max_dist != min_dist:
            potential_roads['normalized_distance'] = 1 - ((potential_roads['distance_to_points'] - min_dist) / (max_dist - min_dist))
        else:
            potential_roads['normalized_distance'] = 1

        # Normalize length (longer is better, as we want main routes)
        if max_length != min_length:
            potential_roads['normalized_length'] = (potential_roads['length'] - min_length) / (max_length - min_length)
        else:
            potential_roads['normalized_length'] = 1

        # Calculate final score (60% comfort, 20% distance, 20% length)
        potential_roads['final_score'] = (
            0.6 * potential_roads['normalized_comfort'] +
            0.2 * potential_roads['normalized_distance'] +
            0.2 * potential_roads['normalized_length']
        )

        # Get the top routes
        comfortable_routes = potential_roads.sort_values(
            by='final_score',
            ascending=False
        ).head(num_routes)

        print(f"Selected {len(comfortable_routes)} comfortable routes")
        
        # Print some details about the selected routes
        for idx, route in comfortable_routes.iterrows():
            print(f"Route {idx}: Comfort Score={route['comfort_score']:.2f}, "
                  f"Final Score={route['final_score']:.2f}, "
                  f"Name={route['NAME'] if 'NAME' in route else 'Unnamed'}")

        return comfortable_routes

    except Exception as e:
        print(f"Error finding comfortable routes: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Function to build a networkx graph from a GeoDataFrame of LineStrings
def build_graph(gdf):
    G = nx.Graph()
    gdf['length'] = gdf.geometry.length # Ensure length column exists
    
    node_id_counter = 0
    nodes = {} # Dictionary to store Point -> node_id mapping
    node_coords = {} # Dictionary to store node_id -> coordinates mapping

    # Add nodes for endpoints and intersections
    for idx, road in gdf.iterrows():
        start = road.geometry.coords[0]
        end = road.geometry.coords[-1]
        
        start_point = Point(start)
        end_point = Point(end)

        start_node_id = nodes.setdefault(start_point.wkt, node_id_counter)
        if start_node_id == node_id_counter:
            node_coords[node_id_counter] = start
            node_id_counter += 1
            
        end_node_id = nodes.setdefault(end_point.wkt, node_id_counter)
        if end_node_id == node_id_counter:
            node_coords[node_id_counter] = end
            node_id_counter += 1

        # Add edge with original index and properties
        G.add_edge(start_node_id, end_node_id, index=idx, weight=road['length'], geometry=road.geometry)

    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, nodes, node_coords

# Function to find the nearest node in the graph to a given point
def find_nearest_node(point, nodes, node_coords):
    nearest_node_id = None
    min_dist = float('inf')
    
    target_point = Point(point)

    for node_id, coords in node_coords.items():
        node_point = Point(coords)
        dist = target_point.distance(node_point)
        if dist < min_dist:
            min_dist = dist
            nearest_node_id = node_id
            
    print(f"Nearest node found: {nearest_node_id} at distance {min_dist}")
    return nearest_node_id

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load shapefile data globally on startup
print("\n=== Starting Application ===")
print(f"Current working directory: {os.getcwd()}")
print("Checking for shapefile...")
shapefile_path = 'FinalTest.shp'
if os.path.exists(shapefile_path):
    print(f"Found shapefile at {shapefile_path}")
    gdf = load_shapefile(shapefile_path)
    if gdf is None:
        print("Error: Failed to load shapefile on startup. Application might not function correctly.")
        # Decide if you want the app to run without the data or exit
        # exit() 
    else:
        print("Shapefile loaded successfully on startup.")
else:
    print(f"Error: Shapefile not found at {shapefile_path}")
    gdf = None # Set gdf to None if file doesn't exist
    # exit()

# --- Flask Routes ---

@app.route('/')
def index():
    try:
        return send_from_directory('.', 'ACCESSBikeComfort_Webtool.html')
    except Exception as e:
        print(f"Error serving index: {str(e)}")
        return str(e), 500

@app.route('/bike_comfort_roads.geojson')
def get_bike_comfort_roads():
    if gdf is None:
        error_msg = 'Failed to load road data. Please check server logs.'
        print(error_msg)
        return jsonify({'error': error_msg}), 500
    return jsonify(json.loads(gdf.to_json()))

@app.route('/health')
def health_check():
    """Route to check if server is running and shapefile is loaded"""
    return jsonify({
        'status': 'running',
        'shapefile_loaded': gdf is not None,
        'features_count': len(gdf) if gdf is not None else 0
    })

@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    try:
        data = request.get_json()
        print(f"Received request data: {data}")
        
        start_address = data.get('start_address')
        end_address = data.get('end_address')
        
        if not start_address or not end_address:
            print("Missing start or end address")
            return jsonify({'error': 'Missing start or end address'}), 400
        
        print(f"\nCalculating route from {start_address} to {end_address}")
        
        start_coords = geocode_address(start_address)
        end_coords = geocode_address(end_address)
        
        if not start_coords or not end_coords:
            print("Failed geocoding")
            return jsonify({'error': 'Could not geocode addresses'}), 400
            
        print(f"Start coords: {start_coords}, End coords: {end_coords}")

        start_point = Point(start_coords[1], start_coords[0]) # lng, lat
        end_point = Point(end_coords[1], end_coords[0]) # lng, lat

        # Define search area (e.g., buffer around start and end)
        buffer_dist = 0.05 # ~5km buffer
        search_area = start_point.buffer(buffer_dist).union(end_point.buffer(buffer_dist))
        
        if gdf is None:
             print("Error: GeoDataFrame not loaded")
             return jsonify({'error': 'Road data not loaded.'}), 500

        potential_roads = gdf[gdf.geometry.intersects(search_area)].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Found {len(potential_roads)} potential roads in search area")

        if len(potential_roads) == 0:
            print("No roads found in search area")
            return jsonify({'error': 'No roads found near start/end points.'}), 400

        # Build graph from potential roads
        G, nodes, node_coords = build_graph(potential_roads)

        # Find the graph nodes closest to the actual start and end points
        start_node = find_nearest_node(start_point.coords[0], nodes, node_coords)
        end_node = find_nearest_node(end_point.coords[0], nodes, node_coords)

        if start_node is None or end_node is None:
             print("Could not find nearest nodes in graph")
             return jsonify({'error': 'Could not snap start/end points to road network.'}), 400

        if start_node == end_node:
             print("Start and end nodes are the same")
             return jsonify({'error': 'Start and end points are too close on the road network.'}), 400

        # Calculate the shortest path using networkx (based on length/weight)
        try:
            path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
            print(f"Shortest path found with {len(path_nodes)} nodes")
        except nx.NetworkXNoPath:
            print("No path found between start and end nodes")
            return jsonify({'error': 'No path found between start and end points on the road network.'}), 400
        except nx.NodeNotFound:
             print("Start or end node not found in graph")
             return jsonify({'error': 'Could not find start or end point in the road network graph.'}), 400

        # Get the original indices of the segments in the path
        path_indices = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data and 'index' in edge_data:
                 path_indices.append(edge_data['index'])
            else:
                 print(f"Warning: Could not find edge data or index for edge ({u}, {v})")

        if not path_indices:
            print("Error: No valid path indices found for the shortest route.")
            return jsonify({'error': 'Could not retrieve segment indices for the calculated path.'}), 500
            
        # Get the geometries using the collected indices
        path_geometries_4326 = potential_roads.loc[path_indices, 'geometry']

        if path_geometries_4326.empty:
             print("Error: No geometries found for the path segments using collected indices.")
             return jsonify({'error': 'Could not retrieve path segment geometries.'}), 500

        # --- Accurate Length Calculation --- 
        total_length_m = 0
        final_route_length_miles = 0
        try:
            # Create a GeoSeries with the correct CRS (EPSG:4326)
            path_gs_4326 = gpd.GeoSeries(path_geometries_4326, crs="EPSG:4326")
            
            # Project to a suitable CRS for length calculation (e.g., EPSG:3857)
            target_crs_for_length = "EPSG:3857"
            print(f"Projecting path segments to {target_crs_for_length} for length calculation...")
            path_gs_proj = path_gs_4326.to_crs(target_crs_for_length)
            
            # Calculate length of each projected segment and sum
            total_length_m = path_gs_proj.length.sum()
            final_route_length_miles = total_length_m * 0.000621371 # Convert meters to miles
            print(f"Calculated total length: {total_length_m:.2f} meters -> {final_route_length_miles:.2f} miles")

        except Exception as proj_err:
            print(f"Warning: Could not project for accurate length calculation: {proj_err}. Length might be inaccurate.")
            final_route_length_miles = 0 
        # --- End Accurate Length Calculation ---

        # Combine original (EPSG:4326) geometries for the final route shape
        final_route_geometry = unary_union(path_geometries_4326)
        
        if final_route_geometry.is_empty:
             print("Final route geometry is empty after union")
             return jsonify({'error': 'Could not construct final route geometry.'}), 500

        # Calculate average comfort score for the segments in the path
        route_comfort_scores = potential_roads.loc[path_indices, 'WeightedCo']
        avg_comfort = route_comfort_scores.mean() if not route_comfort_scores.empty else 0
        
        print(f"Route length: {final_route_length_miles:.2f} miles, Avg Comfort: {avg_comfort:.2f}")

        # Create GeoJSON
        route_geojson = gpd.GeoSeries([final_route_geometry], crs="EPSG:4326").__geo_interface__['features'][0]
        route_geojson['properties'] = {
             'comfort_score': avg_comfort,
             'start_address': start_address,
             'end_address': end_address,
             'total_length': final_route_length_miles,
             'road_segments': len(path_indices)
        }

        print("Route calculation successful")
        return jsonify(route_geojson)

    except Exception as e:
        print(f"Error in calculate_route: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/calculate_distance', methods=['POST'])
def get_distance():
    data = request.get_json()
    start_lat = float(data.get('start_lat'))
    start_lon = float(data.get('start_lon'))
    end_lat = float(data.get('end_lat'))
    end_lon = float(data.get('end_lon'))

    distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    return jsonify({'distance': distance})

# --- Main Execution --- 
if __name__ == '__main__':
    if gdf is None:
        print("Exiting: Shapefile could not be loaded.")
        # Optionally, provide a message indicating the app cannot run without data
    else:
        print("Starting Flask server...")
        # Use port 5001 or change as needed
        app.run(debug=True, port=5001)
