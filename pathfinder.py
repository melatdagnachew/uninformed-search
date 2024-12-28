# Given graph
# Graph representation using dictionary

import networkx as nx
import matplotlib.pyplot as plt
import heapq

graph = {
  'Axum': {'Shire': 250, 'Adwa': 100},
  'Adwa': {'Axum': 100, 'Mekelle': 200},
  'Mekelle': {'Adwa': 200, 'Alamata': 190},
  'Alamata': {'Mekelle': 190, 'Woldiya': 120},
  'Woldiya': {'Alamata': 120, 'Dessie': 165},
  'Dessie': {'Woldiya': 165, 'Kombolcha': 10},
  'Kombolcha': {'Dessie': 10, 'Debre Birhan': 280},
  'Debre Birhan': {'Kombolcha': 280, 'Addis Ababa': 130},
  'Addis Ababa': {'Debre Birhan': 130, 'Bahir Dar': 570},
  'Bahir Dar': {'Addis Ababa': 570, 'Gondar': 180},
  'Gondar': {'Bahir Dar': 180, 'Axum': 230},
  'Shire': {'Axum': 250}
}

print("               *-*-*-GRAPH-*-*-*\n")
for key, value in graph.items():  # Printing Graph
    print(key, ' : ', value)

def unweighted_BFS(graph, start, target):
    queue = [(start, [start])]  # No cost tracking
    visited = set()

    while queue:
        (vertex, path) = queue.pop(0)

        if vertex == target:
            return path

        if vertex not in visited:
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None  # No path found


def weighted_BFS(graph, start, target):
    priority_queue = [(0, start, [start])]  # (cumulative cost, current city, path)
    visited = set()

    while priority_queue:
        cost, vertex, path = heapq.heappop(priority_queue)

        if vertex == target:
            return path, cost

        if vertex not in visited:
            visited.add(vertex)
            for neighbor, weight in graph[vertex].items():
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (cost + weight, neighbor, path + [neighbor]))

    return None, float('inf')  # No path found

def DFS(graph, start, target):    # DFS function
    stack = [(start, [start], 0)]  # Added cost tracking
    while stack:               # Uses Stack data structure
        (vertex, path, cost) = stack.pop()
        for neighbor in graph[vertex]:
            if neighbor not in path:
                new_cost = cost + graph[vertex][neighbor]
                if neighbor == target:
                    return path + [neighbor], new_cost
                else:
                    stack.append((neighbor, path + [neighbor], new_cost))  # Track cost

def uninformed_path_finder(cities, roads, start_city, goal_city, strategy):
    """
    Finds a path between start_city and goal_city using the specified strategy.
    
    Parameters:
        cities: List of city names.
        roads: Dictionary with city connections as {city: {connected_city: distance}}.
        start_city: The city to start the journey.
        goal_city: The destination city.
        strategy: The uninformed search strategy to use ('bfs' or 'dfs').

    Returns:
        path: List of cities representing the path from start_city to goal_city.
        cost: Total cost (distance) of the path.
    """
    if strategy == 'unweighted_bfs':
        return unweighted_BFS(roads, start_city, goal_city), None
    elif strategy == 'weighted_bfs':
        return weighted_BFS(roads, start_city, goal_city)
    elif strategy == 'dfs':
        return DFS(roads, start_city, goal_city)


    return None, float('inf')  # No path found

def visualize_graph(graph, path=None):
    """
    Visualizes the road network and optionally highlights a given path.

    Parameters:
        graph: Dictionary representing the graph {city: {connected_city: distance}}.
        path: List of cities representing the path to highlight (optional).
    """
    G = nx.Graph()
    for city, neighbors in graph.items():
        for neighbor, distance in neighbors.items():
            G.add_edge(city, neighbor, weight=distance)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='red', width=2)

    plt.show()
import random

def traverse_all_cities(cities, roads, start_city, strategy):
    """
    Traverses all cities from a start city to all other cities using the given strategy.
    
    Parameters:
        cities: List of city names.
        roads: Dictionary with city connections as {city: {connected_city: distance}}.
        start_city: The city to start the journey.
        strategy: The strategy to use ('bfs' or 'dfs').
    
    Returns:
        path: List of cities representing the traversal path.
        cost: Total cost (distance) of the traversal.
    """
    # Function to traverse using BFS
    def bfs(cities, roads, start):
        visited = set()
        queue = [(start, [start], 0)]  # (current_city, path, total_cost)
        while queue:
            city, path, cost = queue.pop(0)
            if city not in visited:
                visited.add(city)
                for neighbor, distance in roads[city].items():
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor], cost + distance))
        return path, cost
    
    # Function to traverse using DFS
    def dfs(cities, roads, start):
        visited = set()
        stack = [(start, [start], 0)]  # (current_city, path, total_cost)
        while stack:
            city, path, cost = stack.pop()
            if city not in visited:
                visited.add(city)
                for neighbor, distance in roads[city].items():
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], cost + distance))
        return path, cost

    if strategy == 'bfs':
        return bfs(cities, roads, start_city)
    elif strategy == 'dfs':
        return dfs(cities, roads, start_city)
    else:
        raise ValueError("Invalid strategy. Choose 'bfs' or 'dfs'.")

# Example usage:
cities = list(graph.keys())  # List of city names
start_city = random.choice(cities)  # Randomly pick a start city
strategy = 'bfs'  # Use BFS for traversal
path, total_cost = traverse_all_cities(cities, graph, start_city, strategy)

print(f"Traversal Path: {path} with total cost {total_cost}")
def k_shortest_paths(graph, start, target, k):
    """
    Finds the k-shortest paths between start and target cities.

    Parameters:
        graph: Dictionary representing the graph {city: {connected_city: distance}}.
        start: Starting city.
        target: Destination city.
        k: Number of shortest paths to find.

    Returns:
        List of tuples (path, cost), where path is a list of cities and cost is the total distance.
    """
    def dijkstra_all_paths(graph, start, target):
        """Helper function to find all possible paths with costs using a priority queue."""
        queue = [(0, start, [start])]  # (cumulative cost, current city, path)
        paths = []
        visited = set()

        while queue and len(paths) < k:
            cost, city, path = heapq.heappop(queue)

            if (city, tuple(path)) not in visited:
                visited.add((city, tuple(path)))

                if city == target:
                    paths.append((path, cost))

                for neighbor, distance in graph[city].items():
                    if neighbor not in path:  # Avoid cycles
                        heapq.heappush(queue, (cost + distance, neighbor, path + [neighbor]))
        
        return paths

    # Return the k-shortest paths
    return dijkstra_all_paths(graph, start, target)

start = 'Axum'   # Starting point
target = 'Addis Ababa'   # Ending point / Destination
cities = list(graph.keys())
k = 3

unweighted_path, _ = uninformed_path_finder(cities, graph, start, target, 'unweighted_bfs')
print(f"\nThe shortest path from '{start}' to '{target}' using unweighted BFS is: {unweighted_path}")

# Weighted BFS
weighted_path, weighted_cost = uninformed_path_finder(cities, graph, start, target, 'weighted_bfs')
print(f"\nThe shortest path from '{start}' to '{target}' using weighted BFS is: {weighted_path} with cost {weighted_cost}")

# Example usage with DFS
result = k_shortest_paths(graph, start , target , k)
print(f"The {k}-shortest paths from {start } to {target } are:")
for i, (path, cost) in enumerate(result, 1):
    print(f"{i}: Path: {path}, Cost: {cost}")
    dfs_path, dfs_cost = uninformed_path_finder(cities, graph, start, target, 'dfs')
    print(f"\nThe shortest path from '{start}' to '{target}' using DFS is: {dfs_path} with cost {dfs_cost}")


# Visualize graph and Weighted BFS path
visualize_graph(graph, weighted_path)
