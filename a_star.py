from eomaps import Maps
import numpy as np
import pandas as pd
import heapq
import math 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial import Voronoi
import copy
import matplotlib.cm as cm
import requests
import math
import netCDF4 as nc

# Load the waypoints data   
df = pd.read_csv('/Users/beltran/Documents/University Year 4/Capstone Project/Airmap/datasets/waypoints_graph_zones_grouped.csv')
waypoints = {}

# Iterate over rows of df
for index, row in df.iterrows():
    waypoint_id = row['name']
    lat = row['decimal_lat']
    lon = row['decimal_lon']
    fra_zone = row['fra_zone']
    fra_status = row['fra_status']
    fra_airspace = row['fra_airspace']
    altitude = row['altitude']
    waypoints[waypoint_id] = {'coordinates': (float(lat), float(lon)), 'fra_zone': fra_zone, 'fra_status': fra_status, 'fra_airspace': fra_airspace, 'altitude': altitude}

# Load the weather data
ds = nc.Dataset('/Users/beltran/Documents/University Year 4/Capstone Project/Airmap/datasets/2022-01-31.nc', 'r')
wind_u = ds.variables['wind_u'][:]
wind_v = ds.variables['wind_v'][:]
latitudes = ds.variables['lat'][:]
longitudes = ds.variables['lon'][:]
levels = ds.variables['level'][:]
temperature = ds.variables['temperature'][:]

# Sample data: Altitude in feet and corresponding pressure in mbar
altitude_ft = np.array([-5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000,
                        6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 35000, 
                        40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 
                        85000, 90000, 95000, 100000])

pressure_mbar = np.array([1210.23, 1168.55, 1128.03, 1088.66, 1050.41, 1013.25, 977.166, 
                          942.129, 908.117, 875.105, 843.073, 811.996, 781.854, 752.624, 
                          724.285, 696.817, 571.820, 465.633, 376.009, 300.896, 238.423, 
                          187.54, 147.48, 115.97, 91.199, 71.717, 56.397, 44.377, 34.978, 
                          27.615, 21.837, 17.296, 13.721, 10.902])

# Define a function to model the relationship between altitude and pressure
def altitude_pressure_model_and_prediction(altitude_ft, pressure_mbar, prediction_altitude, degree=5):
    coeffs = np.polyfit(altitude_ft, pressure_mbar, degree)
    polynomial = np.poly1d(coeffs)

    # Return the predicted pressure at the given altitude
    return polynomial(prediction_altitude)

# Example usage
print(f"Pressure at 10,000 ft: {altitude_pressure_model_and_prediction(altitude_ft, pressure_mbar, 10000):.2f} mbar")

# Define a function to calculate the Haversine distance between two points
def haversine_distance(p1, p2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert the coordinates to radians
    lat1 = math.radians(p1[0])
    lon1 = math.radians(p1[1])
    lat2 = math.radians(p2[0])
    lon2 = math.radians(p2[1])

    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance
    distance = R * c

    return distance

# Define a function to check if a point is within an ellipse
def is_within_ellipse(point, start, goal, semi_major_axis, semi_minor_axis):
    # Translate the system so the center of the ellipse is at the origin
    center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
    translated_point = (point[0] - center[0], point[1] - center[1])

    # Rotate the system so the ellipse is aligned with the axes
    angle = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    rotated_point = (translated_point[0] * np.cos(angle) + translated_point[1] * np.sin(angle),
                     -translated_point[0] * np.sin(angle) + translated_point[1] * np.cos(angle))

    # Check if the point is within the ellipse
    return (rotated_point[0] / semi_major_axis) ** 2 + (rotated_point[1] / semi_minor_axis) ** 2 <= 1

# Define a function to calculate the angle between two vectors
def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(dot_product / norm_product)

# Define a function to get the FRA zones of a waypoint
def get_fra_zones(waypoints, waypoint_id):
    return set({zone.strip() for zone in waypoints[waypoint_id]['fra_zone'].split(',')})

# Define a function to find paths between two nodes in a graph
def find_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (node, path) = stack.pop()
        for next_node in set(graph[node]['neighbors'].keys()) - set(path):
            if next_node == goal:
                yield path + [next_node]
            else:
                stack.append((next_node, path + [next_node]))

# Define a function to split the altitude string and calculate the midpoint
def split_altitude(altitude):
    # Split the altitude string on the '/' character
    lower, upper = altitude.split('/')

    # Remove the 'FL' prefix and convert to integers
    lower = int(lower.replace('FL', ''))
    upper = int(upper.replace('FL', ''))

    # Calculate the midpoint of the altitude range
    midpoint = ((lower + upper) / 2)  * 100

    return midpoint

# Define a function to get the wind data at a specific location
def get_weather_data(lat, lon, level):
    # Open a NetCDF file
    # Find the index of the closest latitude, longitude, and level to the ones specified
    lat_index = (np.abs(latitudes - lat)).argmin()
    lon_index = (np.abs(longitudes - lon)).argmin()
    level_index = (np.abs(levels - level)).argmin()

    # Access the wind components at the specified coordinates
    wind_u_value = wind_u[0, level_index, lat_index, lon_index]
    wind_v_value = wind_v[0, level_index, lat_index, lon_index]

    # Access the temperature at the specified coordinates
    temperature_value_k = temperature[0, level_index, lat_index, lon_index]
    temperature_value_c = temperature_value_k - 273.15

    # Calculate the wind speed and direction
    wind_speed = np.sqrt(wind_u_value**2 + wind_v_value**2)
    wind_direction = (270 - (np.arctan2(wind_v_value, wind_u_value) * (180/np.pi))) % 360

    return wind_speed, wind_direction, temperature_value_c

# Define a function to assign wind data to the segments of the path
def assign_wind_data(theGraph, paths, num_segments):
    segment_data = {}
    for path in paths:
        for i in range(len(path) - 1):
            node = path[i]
            neighbor = path[i + 1]
            total_wind_impact = 0
            total_distance = 0
            for j in range(1, num_segments):
                # Calculate intermediate coordinates
                altitude = theGraph[node]['altitude']
                midpoint_altitude = split_altitude(altitude)
                intermediate_coordinates = (
                    theGraph[node]['coordinates'][0] + j * (theGraph[neighbor]['coordinates'][0] - theGraph[node]['coordinates'][0]) / num_segments,
                    theGraph[node]['coordinates'][1] + j * (theGraph[neighbor]['coordinates'][1] - theGraph[node]['coordinates'][1]) / num_segments,
                )
                total_temperature = 0
                total_wind_speed = 0
                total_wind_direction = 0
                total_wind_direction_difference = 0

                # Get weather data at the intermediate coordinates
                wind_speed, wind_direction, temperature = get_weather_data(intermediate_coordinates[0], intermediate_coordinates[1], altitude_pressure_model_and_prediction(altitude_ft, pressure_mbar, midpoint_altitude))

                total_temperature += temperature
                total_wind_speed += wind_speed
                total_wind_direction += wind_direction

                # Calculate the direction of travel along the segment
                travel_direction = math.atan2(theGraph[neighbor]['coordinates'][1] - theGraph[node]['coordinates'][1], theGraph[neighbor]['coordinates'][0] - theGraph[node]['coordinates'][0])
                # Calculate the difference between the wind direction and the travel direction
                wind_direction_difference = math.radians(wind_direction) - travel_direction
                # Calculate the component of the wind speed in the direction of travel
                wind_speed_component = wind_speed * math.cos(wind_direction_difference)
                # Calculate the impact of the wind on the segment
                wind_impact = wind_speed_component
                # Add wind impact to total wind impact
                total_wind_impact += wind_impact
                # Calculate distance to the next node
                distance_to_next = haversine_distance(intermediate_coordinates, theGraph[neighbor]['coordinates'])
                # Add distance to total distance
                total_distance += distance_to_next
                total_wind_direction_difference += wind_direction_difference
            # Calculate average wind direction difference
            avg_wind_direction_difference = total_wind_direction_difference / num_segments if num_segments > 0 else 0
            # Calculate average wind impact over the sum of segments
            avg_wind_impact = total_wind_impact / total_distance if total_distance > 0 else 0
            avg_wind_speed = total_wind_speed / num_segments
            avg_temperature = total_temperature / num_segments
            avg_wind_direction = total_wind_direction / num_segments
            # Assign average wind impact to the segment
            segment_data[(node, neighbor)] = {
                'distance': total_distance,
                'wind_impact': avg_wind_impact,
                'avg_wind_speed': avg_wind_speed,
                'avg_wind_direction': avg_wind_direction_difference,
                'avg_temperature': avg_temperature,
            }
    max_distance = max(segment_data[segment]['distance'] for segment in segment_data.keys())

    for segment in segment_data.keys():
        segment_data[segment]['normalized_distance'] = segment_data[segment]['distance'] / max_distance

    return segment_data

# Define a function to get the neighbors of a node in different fra zones.
def get_neighbors(waypoints, current_node, current_node_fra_zones, goal_node, goal_node_fra_zones, start_node, theGraph, visited_nodes, nodes_to_visit, a, b):
    # Sort waypoints dictionary
    sorted_waypoints = dict(sorted(waypoints.items()))
    # Initialize neighbors list, priority queue, and neighbor counts dictionary
    neighbors = [] 
    priority_queue = []
    neighbor_counts = {}

    # Get eligible nodes based on certain conditions
    eligible_nodes = [node for node in waypoints if node != goal_node and waypoints[node]['fra_status'] in ['EX', 'X', 'E'] and any(zone.strip() in goal_node_fra_zones for zone in waypoints[node]['fra_zone'].split(',')) and set(waypoints[node]['fra_zone'].split(',')) != set(goal_node_fra_zones)]
    # Get the node with minimum distance
    min_distance_node = min(eligible_nodes, key=lambda node: haversine_distance(waypoints[node]['coordinates'], waypoints[start_node]['coordinates']) + haversine_distance(waypoints[node]['coordinates'], waypoints[goal_node]['coordinates']))
    
    # Iterate over sorted waypoints
    for waypoint_id, waypoint_data in sorted_waypoints.items():
        # Get fra zones for the waypoint
        waypoint_fra_zones = get_fra_zones(waypoints, waypoint_id)
        # Check certain conditions to continue the loop
        if waypoint_id != current_node and any(fra_zone in current_node_fra_zones for fra_zone in waypoint_fra_zones) and current_node_fra_zones != waypoint_fra_zones:
            if any(waypoint_id in set(theGraph[node]['neighbors']) for node in theGraph):
                continue
            # Calculate distances
            dist_to_goal = haversine_distance(waypoint_data['coordinates'], waypoints[goal_node]['coordinates'])
            current_to_goal_distance = haversine_distance(waypoints[current_node]['coordinates'], waypoints[goal_node]['coordinates'])
            # Check if the waypoint is within an ellipse
            if is_within_ellipse(waypoint_data['coordinates'], waypoints[start_node]['coordinates'], waypoints[goal_node]['coordinates'], a, b):
                # Check fra status and distance to goal
                if waypoint_data['fra_status'] in ['EX', 'X', 'E'] and dist_to_goal < current_to_goal_distance:
                    # Calculate vectors and angle
                    vector_to_best_from_current = np.array(waypoints[min_distance_node]['coordinates']) - np.array(waypoints[current_node]['coordinates'])
                    vector_to_best_waypoint_id = np.array(waypoints[min_distance_node]['coordinates']) - np.array(waypoints[waypoint_id]['coordinates'])
                    angle = angle_between(vector_to_best_from_current, vector_to_best_waypoint_id)
                    # Push to priority queue
                    heapq.heappush(priority_queue, (angle, waypoint_id))
                
    # Process priority queue until it's empty or we have enough neighbors
    while priority_queue and len(neighbors) < 4:
        # Pop from priority queue
        angle, waypoint_id = heapq.heappop(priority_queue)
        waypoint_data = waypoints[waypoint_id]
        # Calculate distance
        dist = haversine_distance(waypoints[current_node]['coordinates'], waypoint_data['coordinates'])
        # Update theGraph and neighbors list
        theGraph[current_node]['neighbors'][waypoint_id] = dist
        neighbors.append(waypoint_id)
        # Update neighbor counts
        if waypoint_id not in neighbor_counts:
            neighbor_counts[waypoint_id] = 1
        else:
            neighbor_counts[waypoint_id] += 1
        # If waypoint_id is not visited, push it to nodes_to_visit
        if waypoint_id not in visited_nodes:
            heapq.heappush(nodes_to_visit, (angle, waypoint_id, visited_nodes + [waypoint_id]))
    # Return neighbors
    return neighbors

# Define a function to graph the same Free Route Airspace (FRA) zone
def graph_same_fra_zone(waypoints, current_node, current_node_fra_zones, goal_node, goal_node_fra_zones, start_node, theGraph, visited_nodes, nodes_to_visit, a, b):
    # Calculate the distance from the start node to the goal node
    dist_to_goal_from_start = haversine_distance(waypoints[start_node]['coordinates'], waypoints[goal_node]['coordinates'])
    # Add the goal node as a neighbor to the start node in the graph with the calculated distance
    theGraph[start_node]['neighbors'][goal_node] = dist_to_goal_from_start
    # Sort the waypoints
    sorted_waypoints = dict(sorted(waypoints.items()))
    # Initialize an empty list for neighbors
    neighbors = [] 
    # Initialize an empty priority queue
    priority_queue = []
    # Initialize an empty dictionary to keep track of the number of neighbors each waypoint has
    neighbor_counts = {}
    # Set the maximum number of shared neighbors
    max_shared_neighbors = 4
    # Identify eligible nodes that are not the goal node, have a FRA status of 'I', and share at least one FRA zone with the goal node
    eligible_nodes = [node for node in waypoints if node != goal_node and waypoints[node]['fra_status'] == 'I' and any(zone.strip() in goal_node_fra_zones for zone in waypoints[node]['fra_zone'].split(','))]
    # Find the node among the eligible nodes that has the minimum sum of distances to the start and goal nodes
    min_distance_node = min(eligible_nodes, key=lambda node: haversine_distance(waypoints[node]['coordinates'], waypoints[start_node]['coordinates']) + haversine_distance(waypoints[node]['coordinates'], waypoints[goal_node]['coordinates']))
    # Iterate over the sorted waypoints
    for waypoint_id, waypoint_data in sorted_waypoints.items():
        # Get the FRA zones of the current waypoint
        waypoint_fra_zones = get_fra_zones(waypoints, waypoint_id)
        # If the current waypoint is not the current node and shares at least one FRA zone with the current node
        if waypoint_id != current_node and any(fra_zone in current_node_fra_zones for fra_zone in waypoint_fra_zones):
            # If the current waypoint has already reached the maximum number of shared neighbors, skip it
            if waypoint_id in neighbor_counts and neighbor_counts[waypoint_id] >= max_shared_neighbors:
                continue
            # Calculate the distance from the current waypoint to the goal node
            dist_to_goal = haversine_distance(waypoint_data['coordinates'], waypoints[goal_node]['coordinates'])
            # Calculate the distance from the current node to the goal node
            current_to_goal_distance = haversine_distance(waypoints[current_node]['coordinates'], waypoints[goal_node]['coordinates'])
            # If the current waypoint is within the ellipse defined by the start and goal nodes
            if is_within_ellipse(waypoint_data['coordinates'], waypoints[start_node]['coordinates'], waypoints[goal_node]['coordinates'], a, b):
                # If the FRA status of the current waypoint is 'I' and its distance to the goal node is less than the distance from the current node to the goal node
                if waypoint_data['fra_status'] == 'I' and dist_to_goal < current_to_goal_distance:
                    # Calculate the vector from the current node to the node with the minimum distance
                    vector_to_best_from_current = np.array(waypoints[min_distance_node]['coordinates']) - np.array(waypoints[current_node]['coordinates'])
                    # Calculate the vector from the current waypoint to the node with the minimum distance
                    vector_to_best_waypoint_id = np.array(waypoints[min_distance_node]['coordinates']) - np.array(waypoints[waypoint_id]['coordinates'])
                    # Calculate the angle between the two vectors
                    angle = angle_between(vector_to_best_from_current, vector_to_best_waypoint_id)
                    # Add the current waypoint to the priority queue with the angle as the priority
                    heapq.heappush(priority_queue, (angle, waypoint_id))
    # While there are waypoints in the priority queue and the number of neighbors is less than 4
    while priority_queue and len(neighbors) < 4:
        # Pop the waypoint with the smallest angle from the priority queue
        angle, waypoint_id = heapq.heappop(priority_queue)
        # Get the data of the popped waypoint
        waypoint_data = waypoints[waypoint_id]
        # Calculate the distance from the current node to the popped waypoint
        dist = haversine_distance(waypoints[current_node]['coordinates'], waypoint_data['coordinates'])
        # Calculate the distance from the popped waypoint to the goal node
        dist_to_goal = haversine_distance(waypoint_data['coordinates'], waypoints[goal_node]['coordinates'])
        # Add the popped waypoint as a neighbor to the current node in the graph with the calculated distance
        theGraph[current_node]['neighbors'][waypoint_id] = dist
        # Add the goal node as a neighbor to the popped waypoint in the graph with the calculated distance
        theGraph[waypoint_id]['neighbors'][goal_node] = dist_to_goal
        # Add the popped waypoint to the list of neighbors
        neighbors.append(waypoint_id)
        # If the popped waypoint is not in the neighbor counts dictionary, add it with a count of 1
        if waypoint_id not in neighbor_counts:
            neighbor_counts[waypoint_id] = 1
        # If the popped waypoint is in the neighbor counts dictionary, increment its count
        else:
            neighbor_counts[waypoint_id] += 1
        # If the popped waypoint has not been visited, add it to the nodes to visit with the angle as the priority and the current list of visited nodes plus the popped waypoint
        if waypoint_id not in visited_nodes:
            heapq.heappush(nodes_to_visit, (angle, waypoint_id, visited_nodes + [waypoint_id]))
    # Return the graph
    return theGraph

# Define a function to update the graph with the goal node and its FRA zones
def update_graph(waypoints, neighbors, goal_node, goal_node_fra_zones, theGraph):
    # Iterate over the neighbors
    for neighbor in neighbors:
        # Get the FRA zones of the current neighbor
        neighbor_fra_zones = get_fra_zones(theGraph, neighbor)
        # If the FRA zones of the current neighbor and the goal node do not overlap, break the loop
        if neighbor_fra_zones.isdisjoint(goal_node_fra_zones):
            break
        else:
            # If there are neighbors
            if neighbors:
                # Iterate over the nodes in the graph
                for node in list(theGraph.keys()):  # Create a copy of the keys
                    # Iterate over the neighbors of the current node
                    for neighbor in list(theGraph[node]['neighbors'].keys()):
                        # Get the FRA zones of the current neighbor
                        neighbor_fra_zones = get_fra_zones(theGraph, neighbor)
                        # If the FRA zones of the current neighbor and the goal node overlap
                        if neighbor_fra_zones.intersection(goal_node_fra_zones):
                            # Calculate the distance from the current neighbor to the goal node
                            dist_to_goal = haversine_distance(theGraph[neighbor]['coordinates'], waypoints[goal_node]['coordinates'])
                            # Add the goal node as a neighbor to the current neighbor in the graph with the calculated distance
                            theGraph[neighbor]['neighbors'][goal_node] = dist_to_goal
    # Return the updated graph
    return theGraph

# Define a function to integrate wind data into the graph
def integrate_wind(theGraph):
    # Iterate over the nodes in the graph
    for node in theGraph.keys():
        # If the current node has any neighbors
        if theGraph[node]['neighbors']:  
            # Get the coordinates of the current node
            coordinates = theGraph[node]['coordinates']
            # Get the altitude of the current node
            altitude = theGraph[node]['altitude']
            # Calculate the midpoint altitude
            midpoint_altitude = split_altitude(altitude)
            # Get the wind speed, wind direction, and temperature at the current node's coordinates and altitude
            wind_speed, wind_direction, temperature = get_weather_data(coordinates[0], coordinates[1], altitude_pressure_model_and_prediction(altitude_ft, pressure_mbar, midpoint_altitude))  
            # Store the wind speed in the current node's dictionary
            theGraph[node]['wind_speed'] = wind_speed  
            # Store the wind direction in the current node's dictionary
            theGraph[node]['wind_direction'] = wind_direction  
            # Store the temperature in the current node's dictionary
            theGraph[node]['temperature'] = temperature  
    # Return the updated graph
    return theGraph

def calculate_fuel_consumption(TAS, base_TAS, base_FCR, k, distance, wind_speed, wind_angle, temperature, reference_temp=15):
    # Calculate the effect of temperature on air density
    rho_actual = 1.225 * (273.15 + reference_temp) / (273.15 + temperature)  # Basic air density adjustment for temperature
    TAS_adjusted = TAS * (math.sqrt(rho_actual / 1.225))
    
    # Calculate ground speed
    wind_effect = wind_speed * math.cos(math.radians(wind_angle))
    GS = TAS_adjusted + wind_effect
    
    # Adjust fuel consumption rate based on TAS
    FCR = base_FCR * (1 + (TAS_adjusted - base_TAS) / base_TAS * k)
    
    # Calculate total fuel consumption
    flight_duration = distance / GS
    total_fuel = flight_duration * FCR
    
    return total_fuel

def total_route_fuel(graph, aircraft_type, paths):
    if aircraft_type == 'small':
        TAS = 150  # knots
        base_TAS = 120  # knots
        base_FCR = 30  # gallons per hour
        k = 0.02
    elif aircraft_type == 'commercial':
        TAS = 475  # knots
        base_TAS = 430  # knots
        base_FCR = 3000  # kg per hour
        k = 0.03
    else:
        raise ValueError("Invalid aircraft type. Choose either 'small' or 'commercial'.")

    path_fuel_consumptions = []
    for path in paths:
        total_fuel_consumption = 0
        for i in range(len(path) - 1):
            location = path[i]
            neighbor = path[i + 1]
            data = graph[location]['neighbors'][neighbor]

            # Calculate fuel for the leg to the neighbor
            fuel = calculate_fuel_consumption(
                TAS=TAS,
                base_TAS=base_TAS,
                base_FCR=base_FCR,
                k=k,
                distance=data['distance'],
                wind_speed=data['avg_wind_speed'],
                wind_angle=data['avg_wind_direction'],
                temperature=data['avg_temperature'],
                reference_temp=15
            )
            total_fuel_consumption += fuel

        path_fuel_consumptions.append((path, total_fuel_consumption))

    # Sort the paths by fuel consumption and return the five with the least consumption
    path_fuel_consumptions.sort(key=lambda x: x[1])
    return path_fuel_consumptions[:5]
