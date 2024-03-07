import numpy as np
import pandas as pd
from ether.util import is_server_node, is_edge_node


class NetworkSimulation:
    def __init__(self, overlay):
        self.overlay = overlay
        self.sorted_nodes = sorted(overlay.nodes, key=lambda x: x.symphony_id)
        node_names = [node.name for node in self.sorted_nodes]
        self.traffic_matrix = pd.DataFrame(np.zeros((len(self.sorted_nodes), len(self.sorted_nodes))),
                                           index=node_names, columns=node_names)


    def simulate_application_traffic(self):
        """
        Simulates sending data from each edge node to the nearest available server node it can find.
        """
        print(f"self.sorted_nodes {self.sorted_nodes}")
        for node in self.sorted_nodes:
            if is_edge_node(node):
                transfer_type, nearest_server_node = self.determine_transfer_type_and_server(node)
                if nearest_server_node:
                    # Update the server node's processing power based on the task assigned
                    self.update_server_workload_quota(nearest_server_node, transfer_type)
                    # Find the path and update the traffic matrix
                    path = self.overlay.find_symphony_path(node, nearest_server_node)
                    self.update_traffic_matrix_for_path(path, transfer_type)


    def determine_transfer_type_and_server(self, edge_node):
        """
        Determines the transfer type and finds the nearest server node based on workload_quota.
        """
        nearest_server_node = self.find_nearest_server_node(edge_node)
        if nearest_server_node:
            # Check the processing power for the nearest server node and determine transfer type
            if self.has_sufficient_workload_quota(nearest_server_node, 90):
                return ("high", nearest_server_node)
            elif self.has_sufficient_workload_quota(nearest_server_node, 40):
                return ("low", nearest_server_node)
            else:
                # Find the next server node with enough processing power for low res
                next_server_node = self.find_next_server_node(edge_node, 40)
                if next_server_node:
                    return ("low", next_server_node)
        print(f"!!! WARNING !!! Not enough processing resources at cloudlets")
        return (None, None)


    def has_sufficient_workload_quota(self, server_node, required_workload_quota):
        return server_node.workload_quota >= required_workload_quota
    

    def find_nearest_server_node(self, edge_node):
        """
        Finds the nearest server node.
        """
        print(f"first server, edge_node {edge_node}")
        linked_nodes = set(edge_node.bridge_links + edge_node.successor_links + edge_node.predecessor_links + edge_node.long_distance_links)
        for neighbor in linked_nodes:
            if is_server_node(neighbor):
                return neighbor
        
        if edge_node.successor_links:
            first_successor = edge_node.successor_links[0]
            return self.find_nearest_server_node(first_successor)


    def find_next_server_node(self, edge_node, required_workload_quota):
        """
        Finds the next available server node with at least the required processing power.
        """
        print(f"next server, edge_node {edge_node}")
        for node in self.sorted_nodes[self.sorted_nodes.index(edge_node) + 1:]:
            if is_server_node(node) and self.has_sufficient_workload_quota(node, required_workload_quota):
                return node
        return None


    def update_traffic_matrix_for_path(self, path, transfer_type):
        """
        Updates the traffic matrix based on the path taken for data transfer, considering
        2x traffic for intermediate nodes.
        """
        # Map transfer types to their sizes
        transfer_sizes = {"high": 90, "low": 40}
        transfer_size = transfer_sizes.get(transfer_type.lower(), 0)
        
        for i in range(len(path) - 1):
            source_node = path[i]
            destination_node = path[i + 1]
            # Double the traffic size for intermediate nodes
            traffic_multiplier = 2 if i > 0 and i < len(path) - 2 else 1
            self.traffic_matrix.at[source_node.name, destination_node.name] += transfer_size * traffic_multiplier
            print(f"{source_node.name}, {destination_node.name}: {transfer_size * traffic_multiplier}")


    def update_server_workload_quota(self, server_node, transfer_type):
        """
        Updates the server node's processing power based on the resolution of the task assigned.
        """
        transfer_sizes = {"high": 90, "low": 40}
        workload_quota_used = transfer_sizes.get(transfer_type.lower(), 0)
        server_node.workload_quota -= workload_quota_used


    def print_traffic_matrix(self):
        print(self.traffic_matrix)
