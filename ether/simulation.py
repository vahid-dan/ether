import numpy as np
from ether.util import is_server_node, is_edge_node


class NetworkSimulation:
    def __init__(self, overlay):
        self.overlay = overlay
        self.sorted_nodes = sorted(overlay.nodes, key=lambda x: x.symphony_id)
        self.traffic_matrix = np.zeros((len(self.sorted_nodes), len(self.sorted_nodes)))


    def simulate_application_traffic(self):
        """
        Simulates sending 1 GB of data per day from each edge node to the nearest server node it can find.
        """
        for node in self.sorted_nodes:
            print(node)
            if is_edge_node(node):
                nearest_server_node = self.find_nearest_server_node(node)
                print(f"nearest_server_node {nearest_server_node}")
                if nearest_server_node:
                    path = self.overlay.find_symphony_path(node, nearest_server_node)
                    print(f"path {path}")
                    self.update_traffic_matrix_for_path(path)


    def find_nearest_server_node(self, edge_node):
        """
        Finds the nearest server node.
        """
        linked_nodes = set(edge_node.bridge_links + edge_node.successor_links + edge_node.predecessor_links + edge_node.long_distance_links)
        for neighbor in linked_nodes:
            if is_server_node(neighbor):
                return neighbor
        
        if edge_node.successor_links:
            first_successor = edge_node.successor_links[0]
            return self.find_nearest_server_node(first_successor)


    def update_traffic_matrix_for_path(self, path):
        """
        Updates the traffic matrix based on the path taken for data transfer.
        """
        for i in range(len(path) - 1):
            source_index = self.sorted_nodes.index(path[i])
            destination_index = self.sorted_nodes.index(path[i + 1])
            print(f"source_index {source_index} destination_index {destination_index}")
            self.traffic_matrix[source_index, destination_index] += 1
