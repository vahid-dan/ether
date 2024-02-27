import hashlib
import numpy as np
import random
import time
from typing import List
import math

from ether.core import Node
from ether.util import is_edge_node, is_server_node
from ether.topology import Topology
from ether.link_selection import get_potential_targets_randomly, get_potential_targets_from_neighborhood, decide_topsis



class SymphonyOverlay:
    def __init__(self, nodes: List[Node], seed=None):
        self.nodes = nodes
        for node in self.nodes:
            node.symphony_id = self.generate_symphony_id(node, seed)
        self.initialize_links()


    def initialize_links(self):
        for node in self.nodes:
            # Initialize successor links if not already present
            if not hasattr(node, 'successor_links'):
                node.successor_links = []
            # Initialize predecessor links if not already present
            if not hasattr(node, 'predecessor_links'):
                node.predecessor_links = []
            # Initialize long distance links if not already present
            if not hasattr(node, 'long_distance_links'):
                node.long_distance_links = []


    @staticmethod
    def generate_symphony_id(node: Node, seed=None) -> float:
        if seed is not None:
            random.seed(seed)
            seed = str(seed) + node.name
        else:
            seed = f"{node.name}_{time.time()}_{random.randint(0, 1e6)}"
        
        hash_result = hashlib.sha256(seed.encode()).digest()
        int_value = int.from_bytes(hash_result, byteorder='big')
        max_int_value_for_sha256 = 2 ** 256 - 1
        unique_id = int_value / max_int_value_for_sha256
        return unique_id


    def set_successor_links(self):
        """
        Set both successor and predecessor links for this node based on the Symphony ring structure.
        """
        sorted_nodes = sorted(self.nodes, key=lambda x: x.symphony_id)
        for i, node in enumerate(sorted_nodes):
            num_nodes = len(sorted_nodes)
            node.successor_links = [sorted_nodes[(i + 1) % num_nodes], sorted_nodes[(i + 2) % num_nodes]]
            node.predecessor_links = [sorted_nodes[(i - 1) % num_nodes], sorted_nodes[(i - 2) % num_nodes]]


    def set_long_distance_links(self, topology: Topology, target_selection_strategy: str = 'harmonic', decision_method: str = 'topsis', weights=np.array([1, 1]), is_benefit=np.array([False, False])):
        """
        Randomly select a node and attempt to assign a single link, repeating for a specified number of iterations.
        """
        sorted_nodes = sorted(self.nodes, key=lambda x: x.symphony_id)
        num_nodes = len(sorted_nodes)
        selection_size_factor = round(math.log(num_nodes))
        max_num_links = round(math.log(num_nodes))
        servers_max_num_links = max_num_links * 8
        total_iterations = int(num_nodes * max_num_links * selection_size_factor / 2)
        max_total_links = float('inf')
        total_links_created = 0

        for _ in range(total_iterations):
            # Check if the network has reached the maximum total number of links
            if total_links_created >= max_total_links:
                print("Reached the maximum total number of links for the network.")
                break

            # Randomly select a node
            node_index = np.random.randint(0, num_nodes)
            node = sorted_nodes[node_index]

            # Skip nodes that have reached the max number of links
            if is_server_node(node):
                if len(node.long_distance_links) >= servers_max_num_links:
                    continue
            elif len(node.long_distance_links) >= max_num_links:
                continue

            # Select potential targets based on the specified strategy
            if target_selection_strategy == 'neighborhood':
                potential_targets = get_potential_targets_from_neighborhood(sorted_nodes, node, num_nodes, selection_size_factor, max_num_links, servers_max_num_links)
            elif target_selection_strategy == 'harmonic':
                potential_targets = get_potential_targets_randomly(sorted_nodes, node, num_nodes, selection_size_factor, max_num_links, servers_max_num_links)
            else:
                raise ValueError(f"Unknown target selection strategy: {target_selection_strategy}")

            if potential_targets:
                # Decide on the best target based on the specified decision method
                if decision_method == 'topsis':
                    criteria = [[topology.latency(node, target, use_coordinates=True), target.cell_cost] for target in potential_targets]
                    criteria_matrix = np.array(criteria)
                    best_target_index = decide_topsis(criteria_matrix, weights, is_benefit)
                    best_target = potential_targets[best_target_index]
                elif decision_method == 'random':
                    best_target_index = 0  # The first choice in the list of potential targets
                    best_target = potential_targets[best_target_index]
                else:
                    raise ValueError(f"Unknown decision method: {decision_method}")
                
                # Establish links with randomly selected targets
                node.long_distance_links.append(best_target)
                best_target.long_distance_links.append(node)  # Ensure bidirectional links
                total_links_created += 1
                if total_links_created >= max_total_links:
                    print(f"Reached max_total_links {max_total_links}")
                print(f"Link {node} -----> {best_target}")

        print(f"total_links_created {total_links_created}")
        return


    def find_closest_clockwise_node(self, current_node, destination_node):
        """
        Find the node closest to the destination in a clockwise direction,
        handling the wrap-around of the ring.
        """
        all_links = current_node.successor_links + current_node.predecessor_links + current_node.long_distance_links
        sorted_links = sorted(all_links, key=lambda node: node.symphony_id)

        # Wrap-around case
        if current_node.symphony_id > destination_node.symphony_id:
            closest_node_after_wrap = None

            # Find the node closest to the start of the ring, but before the destination_id
            for node in sorted_links:
                if node.symphony_id <= destination_node.symphony_id:
                    if closest_node_after_wrap is None or node.symphony_id > closest_node_after_wrap.symphony_id:
                        closest_node_after_wrap = node

            # If a node is found after the wrap, return it
            if closest_node_after_wrap:
                return closest_node_after_wrap

            # Otherwise, return the node closest to the end of the ring
            closest_node_to_end = None
            for node in sorted_links:
                if node.symphony_id > current_node.symphony_id:
                    if closest_node_to_end is None or node.symphony_id > closest_node_to_end.symphony_id:
                        closest_node_to_end = node

            return closest_node_to_end if closest_node_to_end else sorted_links[0]

        # Normal case: destination is ahead in the ring
        closest_node = None
        for node in sorted_links:
            if node.symphony_id > current_node.symphony_id and (closest_node is None or node.symphony_id <= destination_node.symphony_id and node.symphony_id > closest_node.symphony_id):
                closest_node = node

        if closest_node:
            return closest_node

        # Fallback: if no node is found, return the first node in the sorted list
        return sorted_links[0]


    def find_symphony_path(self, start_node, destination_node):
        """
        Find the path from this node to the destination_node using Symphony routing.
        
        :param destination_node: The end node of the path
        :param all_nodes: A list of all nodes in the Symphony network
        :return: A list of nodes representing the path from this node to the destination_node
        """
        path = [start_node]
        current_node = start_node

        while current_node != destination_node:
            next_node = self.find_closest_clockwise_node(current_node, destination_node)
            path.append(next_node)
            current_node = next_node

        return path
    

    def assign_cell_costs(self, percentage):
        # Filter edge nodes using the is_edge_node function
        edge_nodes = [node for node in self.nodes if is_edge_node(node)]
        
        # Calculate number of nodes to assign random cost
        num_metered_edge_nodes = int(len(edge_nodes) * (percentage / 100.0))
        
        # Randomly select nodes for random cost assignment
        metered_edge_nodes = random.sample(edge_nodes, num_metered_edge_nodes) if edge_nodes else []
        
        # Assign random cost to selected nodes and 0 to others
        for node in self.nodes:
            if node in metered_edge_nodes:
                # node.cell_cost = random.uniform(0, 1)
                node.cell_cost = 1
            else:
                node.cell_cost = 0


    def calculate_total_long_distance_metrics(self, topology):
        total_latency = 0
        total_cost = 0
        for node in self.nodes:
            for linked_node in node.long_distance_links:
                total_latency += topology.latency(node, linked_node, use_coordinates=False)
                total_cost += linked_node.cell_cost
        return total_latency, total_cost
