import numpy as np
import pandas as pd
from ether.util import is_server_node, is_edge_node, is_cloud_server_node
from ether.link_selection import decide_topsis
from ether.topology import Topology


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


    def simulate_application_traffic(self,
                                     topology: Topology,
                                     select_server_decision_method='topsis',
                                     select_server_weights=np.array([1, 1]),
                                     select_server_is_benefit=np.array([True, False])):
        """
        Simulates sending data from each edge node to the nearest available server node it can find.
        """
        print(f"self.sorted_nodes {self.sorted_nodes}")
        for node in self.sorted_nodes:
            if is_edge_node(node):
                print(f"----- {node} -----")
                transfer_type, nearest_server_node = self.determine_transfer_type_and_server(topology=topology,
                                                                                             node=node,
                                                                                             select_server_decision_method=select_server_decision_method,
                                                                                             select_server_weights=select_server_weights,
                                                                                             select_server_is_benefit=select_server_is_benefit)
                if nearest_server_node:
                    # Update the server node's processing power based on the task assigned
                    self.update_server_workload_quota(nearest_server_node, transfer_type)
                    # Find the path and update the traffic matrix
                    path = self.overlay.find_symphony_path(node, nearest_server_node)
                    self.update_traffic_matrix_for_path(path, transfer_type)
                    self.assign_final_step_task(nearest_server_node)


    def determine_transfer_type_and_server(self,
                                           topology: Topology,
                                           node,
                                           select_server_decision_method,
                                           select_server_weights,
                                           select_server_is_benefit):
        """
        Determines the transfer type and finds the nearest server node based on workload_quota.
        """
        nearest_server_node = self.find_nearest_server_node(topology,
                                                            node,
                                                            select_server_decision_method,
                                                            select_server_weights,
                                                            select_server_is_benefit)
        if nearest_server_node:
            print(f"nearest_server_node {nearest_server_node}")
            # Check the processing power for the nearest server node and determine transfer type
            if self.has_sufficient_workload_quota(nearest_server_node, self.workload_sizes["high"]):
                print("first server found, high")
                return ("high", nearest_server_node)
            elif self.has_sufficient_workload_quota(nearest_server_node, self.workload_sizes["low"]):
                print("first server found, low")
                return ("low", nearest_server_node)
            else:
                # Find the next server node with enough processing power for low res
                print("not enough quota in the first server, looking for the next.")
                next_server_node = self.find_next_server_node(node, self.workload_sizes["low"])
                if next_server_node:
                    print("next server found")
                    return ("low", next_server_node)
        print(f"!!! WARNING !!! Not enough processing resources at cloudlets")
        return (None, None)


    def has_sufficient_workload_quota(self, server_node, required_workload_quota):
        print(f"{server_node} quota {server_node.workload_quota}")
        return server_node.workload_quota >= required_workload_quota
    

    def find_nearest_server_node(self,
                                 topology: Topology,
                                 node,
                                 select_server_decision_method,
                                 select_server_weights,
                                 select_server_is_benefit):
        """
        Finds the nearest server node using a specified decision method.
        """
        print(f"node {node} aiming for first server")
        neighbors = set([node] + node.bridge_links + node.successor_links + node.predecessor_links + node.long_distance_links)
        print(f"neighbors {neighbors}")

        # Filter neighbors to include only server nodes
        server_neighbors = [neighbor for neighbor in neighbors if is_server_node(neighbor)]

        if server_neighbors:
            # If there are server nodes among the neighbors, decide on the nearest based on the decision method
            if select_server_decision_method == 'topsis':
                # Example criteria: [(latency, processing_power), ...]
                criteria = [(topology.latency(node, server, use_coordinates=True), server.processing_power) for server in server_neighbors]
                criteria_matrix = np.array(criteria)
                best_target_index = decide_topsis(criteria_matrix, select_server_weights, select_server_is_benefit)
                best_server = server_neighbors[best_target_index]
                print(f"Selected server using TOPSIS: {best_server}")
                return best_server
            elif select_server_decision_method == 'random':
                # Randomly select a server from the neighbors
                best_server = np.random.choice(server_neighbors)
                print(f"Selected server randomly: {best_server}")
                return best_server
            else:
                raise ValueError(f"Unknown decision method: {select_server_decision_method}")
        else:
            print("No server node among immediate neighbors. Expanding search...")

            # Expand search beyond immediate neighbors
            # Check if the node is a pendant node and move to its connected switch node
            if node.role == 'pendant':
                print("edge node pendant")
                return self.find_nearest_server_node(topology,
                                                     node.bridge_links[0],
                                                     select_server_decision_method,
                                                     select_server_weights,
                                                     select_server_is_benefit)
                    
            elif node.successor_links:
                first_successor = node.successor_links[0]
                print("move to first successor {first_successor}")
                return self.find_nearest_server_node(topology,
                                                     first_successor,
                                                     select_server_decision_method,
                                                     select_server_weights,
                                                     select_server_is_benefit)

        # Fallback if no server found in neighbors and expanded search is not implemented
        print("!!! WARNING !!! No suitable server found")
        return None


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


    def calculate_and_update_node_costs(self, cost_per_unit):
        """
        Calculates and updates the cost for each node based on the application traffic matrix
        and a given cost per unit of data transferred.
        """
        for source_node_name in self.traffic_matrix.index:
            for destination_node_name in self.traffic_matrix.columns:
                data_transferred = self.traffic_matrix.at[source_node_name, destination_node_name]
                source_node = next((node for node in self.sorted_nodes if node.name == source_node_name), None)
                
                if source_node and getattr(source_node, 'is_metered', False):                  
                    source_node.cell_cost += data_transferred * cost_per_unit


    def print_node_costs(self):
        """
        Prints the costs for each node.
        """
        for node in self.sorted_nodes:
            if hasattr(node, 'cell_cost') and node.cell_cost > 0:
                print(f"{node.name} cost: {node.cell_cost}")
