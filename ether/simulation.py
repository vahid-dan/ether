import numpy as np
import pandas as pd
from ether.util import is_server_node, is_edge_node, is_cloud_server_node


class NetworkSimulation:
    # Define data transfer sizes for high and low resolution
    transfer_sizes = {"high": 90, "low": 40, "final_step": 5}
    
    # Define workload sizes for processing high and low resolution data
    workload_sizes = {"high": 90, "low": 40, "final_step": 5}

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
                print(f"----- {node} -----")
                transfer_type, nearest_server_node = self.determine_transfer_type_and_server(node)
                if nearest_server_node:
                    # Update the server node's processing power based on the task assigned
                    self.update_server_workload_quota(nearest_server_node, transfer_type)
                    # Find the path and update the traffic matrix
                    path = self.overlay.find_symphony_path(node, nearest_server_node)
                    self.update_traffic_matrix_for_path(path, transfer_type)
                    self.assign_final_step_task(nearest_server_node)


    def determine_transfer_type_and_server(self, edge_node):
        """
        Determines the transfer type and finds the nearest server node based on workload_quota.
        """
        nearest_server_node = self.find_nearest_server_node(edge_node)
        if nearest_server_node:
            print(f"nearest_server_node {nearest_server_node}")
            # Check the processing power for the nearest server node and determine transfer type
            if self.has_sufficient_workload_quota(nearest_server_node, self.workload_sizes["high"]):
                print("high")
                return ("high", nearest_server_node)
            elif self.has_sufficient_workload_quota(nearest_server_node, self.workload_sizes["low"]):
                print("low")
                return ("low", nearest_server_node)
            else:
                # Find the next server node with enough processing power for low res
                print("not enough quota in the first server, looking for the next.")
                next_server_node = self.find_next_server_node(edge_node, self.workload_sizes["low"])
                if next_server_node:
                    print("next server found")
                    return ("low", next_server_node)
        print(f"!!! WARNING !!! Not enough processing resources at cloudlets")
        return (None, None)


    def has_sufficient_workload_quota(self, server_node, required_workload_quota):
        print(f"{server_node} quota {server_node.workload_quota}")
        return server_node.workload_quota >= required_workload_quota
    

    def find_nearest_server_node(self, edge_node):
        """
        Finds the nearest server node.
        """
        print(f"edge_node {edge_node} aiming for first server")
        neighbors = set(edge_node.bridge_links + edge_node.successor_links + edge_node.predecessor_links + edge_node.long_distance_links)
        print(f"neighbors {neighbors}")
        for neighbor in neighbors:
            if is_server_node(neighbor):
                print(f"found server in neighbors {neighbor}")
                return neighbor
        print("no neighbor server")

        # Check if the node is a pendant node and move to its connected switch node
        if edge_node.role == 'pendant':
            print("edge node pendant")
            return self.find_nearest_server_node(edge_node.bridge_links[0])
        
        if edge_node.successor_links:
            first_successor = edge_node.successor_links[0]
            print("move to first successor {first_successor}")
            return self.find_nearest_server_node(first_successor)


    def find_next_server_node(self, edge_node, required_workload_quota):
        """
        Finds the next available server node with at least the required processing power.
        """
        print(f"edge_node {edge_node} aiming for next server")
        edge_node_index = self.sorted_nodes.index(edge_node)
        for node in self.sorted_nodes[edge_node_index + 1:]:
            if is_server_node(node) and self.has_sufficient_workload_quota(node, required_workload_quota):
                print(f"next server found {node}")
                return node
        return None


    def update_traffic_matrix_for_path(self, path, transfer_type):
        """
        Updates the traffic matrix based on the path taken for data transfer, considering
        2x traffic (incoming + outgoing) for intermediate nodes.
        """
        transfer_size = self.transfer_sizes.get(transfer_type.lower(), 0)
        
        for i in range(len(path) - 1):
            source_node = path[i]
            destination_node = path[i + 1]
            # Double the traffic size for intermediate nodes
            traffic_multiplier = 2 if i > 0 and i < len(path) - 2 else 1
            self.traffic_matrix.at[source_node.name, destination_node.name] += transfer_size * traffic_multiplier
            print(f"{source_node.name}, {destination_node.name}: {transfer_size * traffic_multiplier}")


    def update_server_workload_quota(self, server_node, transfer_type):
        """
        Updates the cloudlet's workload quota based on the resolution of the task assigned,
        affecting all servers within the same cloudlet.
        """
        workload_quota_used = self.workload_sizes.get(transfer_type.lower(), 0)
        if is_cloud_server_node(server_node):
            servers = [node for node in self.sorted_nodes if is_cloud_server_node(node)]
        else:
            servers = [node for node in self.sorted_nodes if is_server_node(node) and node.location_id == server_node.location_id]
        
        for node in servers:
            node.workload_quota -= workload_quota_used


    def assign_final_step_task(self, cloudlet_server_node):
        # Find a suitable cloud server for the final_step task
        cloud_server_node = self.find_suitable_cloud_server()
        if cloud_server_node:
            print(f"{cloud_server_node}")
            self.update_server_workload_quota(cloud_server_node, "final_step")
            path = self.overlay.find_symphony_path(cloudlet_server_node, cloud_server_node)
            self.update_traffic_matrix_for_path(path, "final_step")


    def find_suitable_cloud_server(self):
        # Implement logic to find a cloud server with enough workload quota for the final_step task
        for node in self.sorted_nodes:
            if is_cloud_server_node(node) and self.has_sufficient_workload_quota(node, self.workload_sizes["final_step"]):
                return node
        return None


    def print_traffic_matrix(self):
        print(self.traffic_matrix)
        for node in self.sorted_nodes:
            if hasattr(node, 'workload_quota'):
                print(f"{node} workload_quota {node.workload_quota}")
