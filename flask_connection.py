from flask import Flask, render_template, request
import pandas as pd
import math
from a_star import get_fra_zones, get_neighbors, update_graph, graph_same_fra_zone, find_paths, integrate_wind, assign_wind_data, total_route_fuel
from flask import jsonify
import heapq

app = Flask(__name__)

# Define routes
@app.route('/')

def index():
    return render_template('index.html')

# Load waypoints data
def data_structure():
    df = pd.read_csv('/Users/beltran/Documents/University Year 4/Capstone Project/Airmap/datasets/waypoints_graph_zones_grouped.csv')
    waypoints = {}
    for index, row in df.iterrows():
        waypoint_id = row['name']
        lat = row['decimal_lat']
        lon = row['decimal_lon']
        fra_zone = row['fra_zone']
        fra_status = row['fra_status']
        fra_airspace = row['fra_airspace']
        altitude = row['altitude']
        waypoints[waypoint_id] = {'coordinates': (float(lat), float(lon)), 'fra_zone': fra_zone, 'fra_status': fra_status, 'fra_airspace': fra_airspace, 'altitude': altitude}

    return waypoints

waypoints = data_structure()

@app.route('/calculate_route', methods=['POST'])

# Define the function that will be called when the route is accessed
def get_route():
    # Create a dictionary with the graph structure
    theGraph = {waypoint_id: {'coordinates': waypoint_data['coordinates'], 'fra_status': waypoint_data['fra_status'], 'fra_airspace': waypoint_data['fra_airspace'], 'fra_zone': waypoint_data['fra_zone'], 'altitude': waypoint_data['altitude'], 'neighbors': {}} for waypoint_id, waypoint_data in waypoints.items()}
    print('get_route() is being called')
    # Get form data from the request
    start_node = request.form['start_node']
    goal_node = request.form['goal_node']
    # Perform calculations using your Python functions
    start = waypoints[start_node]['coordinates']
    goal = waypoints[goal_node]['coordinates']
    aircraft_type = request.form['aircraft_type'] 

    # Perform calculations using your Python functions
    nodes_to_visit = [(0, start_node, [])] 

    # Get the FRA zones of the start and goal nodes
    start_node_fra_zones = get_fra_zones(waypoints, start_node)
    goal_node_fra_zones = get_fra_zones(waypoints, goal_node)

    # Calculate an ellipse with the major axis being the distance between the start and goal nodes and the minor axis being a quarter of the major axis
    a = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2) / 2
    b = a / 4

    # Remove the goal node from the neighbors of the goal node
    if goal_node in theGraph[goal_node]['neighbors']:
        del theGraph[goal_node]['neighbors'][goal_node]
    
    # Create loop to find the shortest path between the start and goal nodes in the same FRA zone.
    if not start_node_fra_zones.isdisjoint(goal_node_fra_zones):
        while nodes_to_visit:
            _, current_node, visited_nodes = heapq.heappop(nodes_to_visit)
            current_node_fra_zones = get_fra_zones(waypoints, current_node)
            theGraph = graph_same_fra_zone(waypoints, current_node, current_node_fra_zones, goal_node, goal_node_fra_zones, start_node, theGraph, visited_nodes, nodes_to_visit, a, b)
    
    # Create loop to find the shortest path between the start and goal nodes in different FRA zones.
    while nodes_to_visit:
        _, current_node, visited_nodes = heapq.heappop(nodes_to_visit)
        current_node_fra_zones = get_fra_zones(waypoints, current_node)
        neighbors = get_neighbors(waypoints, current_node, current_node_fra_zones, goal_node, goal_node_fra_zones, start_node, theGraph, visited_nodes, nodes_to_visit, a, b)
        theGraph = update_graph(waypoints, neighbors, goal_node, goal_node_fra_zones, theGraph)

    # Find the five shortest paths between the start and goal nodes
    shortest_paths = list(find_paths(theGraph, start_node, goal_node))
    five_shortest_paths = shortest_paths[:5]
    # Integrate wind data into the graph
    theGraph = integrate_wind(theGraph)
    # Assign wind data to the segments of the five shortest paths
    segments = assign_wind_data(theGraph, five_shortest_paths, num_segments=10) 

    final_graph = {}
    for (node1, node2), data in segments.items():
        if node1 not in final_graph:
            final_graph[node1] = {'neighbors': {}}
        if node2 not in final_graph:
            final_graph[node2] = {'neighbors': {}}
        final_graph[node1]['neighbors'][node2] = {'distance': data['distance'], 'avg_wind_speed': data['avg_wind_speed'], 'avg_wind_direction': data['avg_wind_direction'], 'avg_temperature': data['avg_temperature']}

    # Calculate the total fuel consumption of the five shortest paths
    paths = five_shortest_paths
    paths = total_route_fuel(final_graph, aircraft_type, five_shortest_paths)

    paths_with_coordinates_and_fuel = []

    # Create a list of dictionaries with the paths, fuel consumption, distance, and names of the waypoints
    for path, fuel_consumption in paths:
        path_with_coordinates = [{'name': waypoint, 'coordinates': waypoints[waypoint]['coordinates'], 'fra_zone': waypoints[waypoint]['fra_zone'], 'altitude': waypoints[waypoint]['altitude']} for waypoint in path]
        path_distance = sum(final_graph[path[i]]['neighbors'][path[i+1]]['distance'] for i in range(len(path)-1))  # Calculate the total distance of the path
        path_names = [waypoint for waypoint in path]  # Get the names of the waypoints
        paths_with_coordinates_and_fuel.append({'path': path_with_coordinates, 'fuel_consumption': fuel_consumption, 'distance': path_distance, 'names': path_names})

    # Return the result as a JSON
    return jsonify(paths=paths_with_coordinates_and_fuel)

if __name__ == '__main__':
    app.run(debug=True)
