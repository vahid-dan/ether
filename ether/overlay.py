import hashlib
import numpy as np
import random
import time
import math
from typing import List

from ether.core import Node
from ether.util import harmonic_random_number, topsis, is_edge_node
from ether.topology import Topology


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


    def set_long_distance_links(self, topology: Topology, max_num_links: int = 2, link_selection_method: str = 'topsis', weights=np.array([1, 1]), is_benefit=np.array([False, False]), candidate_list_size_factor: int = 2):
        """
        Set the long-distance links for this node based on the Symphony network properties.
        """
        sorted_nodes = sorted(self.nodes, key=lambda x: x.symphony_id)
        num_nodes = len(sorted_nodes)
        selection_size_factor = 4
        attempt_factor = num_nodes * selection_size_factor
        max_attempts = num_nodes * attempt_factor # Set a limit to prevent infinite loops
        if candidate_list_size_factor > selection_size_factor:
            raise ValueError(f"candidate_list_size_factor {candidate_list_size_factor} is larger than {selection_size_factor} and not accepted.")

        for node in sorted_nodes:
            # Check if the node already has the maximum number of long-distance links
            if len(node.long_distance_links) >= max_num_links:
                print(f"Max number of links reached for {node}.")
                continue

            potential_targets = []
            found_targets = 0
            attempt_counter = 0

            # Try to find potential targets up to a calculated maximum number of attempts
            while found_targets < selection_size_factor * max_num_links and attempt_counter < max_attempts:
                attempt_counter += 1
                # Select a random target based on harmonic distribution
                index = harmonic_random_number(num_nodes) - 1
                potential_target = sorted_nodes[index]

                # Ensure the potential target meets all criteria for selection
                if (len(potential_target.long_distance_links) < max_num_links and
                    potential_target != node and
                    potential_target not in node.successor_links and
                    potential_target not in node.predecessor_links and
                    potential_target not in node.long_distance_links and
                    potential_target not in potential_targets): # Ensure potential target is unique
                    potential_targets.append(potential_target)
                    found_targets += 1

            if attempt_counter == max_attempts:
                print(f"Max number of attempts reached for {node}. Found {len(potential_targets)} unique potential targets.")

            # Check if the number of found targets is less than expected
            if len(potential_targets) < selection_size_factor * max_num_links:
                print(f"Only found {len(potential_targets)} unique potential targets for {node}, but expected {selection_size_factor * max_num_links}.")
            # Apply the TOPSIS method for link selection
            shortlisted_targets_desired_size = min(len(potential_targets), candidate_list_size_factor * max_num_links)
            shortlisted_targets = potential_targets[:shortlisted_targets_desired_size]
            if link_selection_method == 'topsis' and shortlisted_targets:
                criteria = []
                for target in shortlisted_targets:
                    # Calculate link_latency as a criterion for TOPSIS
                    link_latency = topology.latency(node, target, use_coordinates=True)
                    link_cell_cost = node.cell_cost + target.cell_cost
                    criteria.append([link_latency, link_cell_cost])

                criteria_matrix = np.array(criteria)
                # Calculate TOPSIS scores
                scores = topsis(criteria_matrix, weights, is_benefit)
                # Select the top targets based on scores
                finalist_targets = [shortlisted_targets[i] for i in np.argsort(scores)[::-1][:max_num_links]]

                # Establish links with the selected targets
                for target in finalist_targets:
                    if len(node.long_distance_links) < max_num_links and len(target.long_distance_links) < max_num_links:
                        node.long_distance_links.append(target)
                        target.long_distance_links.append(node) # Ensure bidirectional links

            # Use a random method for link selection
            elif link_selection_method == 'random':
                for target in potential_targets:
                    # Establish links with randomly selected targets, respecting max_num_links
                    if len(node.long_distance_links) < max_num_links and len(target.long_distance_links) < max_num_links:
                        node.long_distance_links.append(target)
                        target.long_distance_links.append(node) # Ensure bidirectional links
                    elif len(node.long_distance_links) >= max_num_links:
                        break


    def find_closest_clockwise_node(self, current_node, destination_id):
        """
        Find the node closest to the destination in a clockwise direction,
        handling the wrap-around of the ring.
        """
        all_links = current_node.successor_links + current_node.predecessor_links + current_node.long_distance_links
        sorted_links = sorted(all_links, key=lambda node: node.symphony_id)

        # Wrap-around case
        if self.symphony_id > destination_id:
            closest_node_after_wrap = None

            # Find the node closest to the start of the ring, but before the destination_id
            for node in sorted_links:
                if node.symphony_id <= destination_id:
                    if closest_node_after_wrap is None or node.symphony_id > closest_node_after_wrap.symphony_id:
                        closest_node_after_wrap = node

            # If a node is found after the wrap, return it
            if closest_node_after_wrap:
                return closest_node_after_wrap

            # Otherwise, return the node closest to the end of the ring
            closest_node_to_end = None
            for node in sorted_links:
                if node.symphony_id > self.symphony_id:
                    if closest_node_to_end is None or node.symphony_id > closest_node_to_end.symphony_id:
                        closest_node_to_end = node

            return closest_node_to_end if closest_node_to_end else sorted_links[0]

        # Normal case: destination is ahead in the ring
        closest_node = None
        for node in sorted_links:
            if node.symphony_id > self.symphony_id and (closest_node is None or node.symphony_id <= destination_id and node.symphony_id > closest_node.symphony_id):
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
            next_node = self.find_closest_clockwise_node(current_node, destination_node.symphony_id)
            path.append(next_node)
            current_node = next_node

        return path
    

    def assign_cell_costs(self, percentage):
        # Filter edge nodes using the is_edge_node function
        edge_nodes = [node for node in self.nodes if is_edge_node(node)]
        
        # Calculate number of nodes to assign random cost
        num_random_cost = int(len(edge_nodes) * (percentage / 100.0))
        
        # Randomly select nodes for random cost assignment
        metered_nodes = random.sample(edge_nodes, num_random_cost) if edge_nodes else []
        
        # Assign random cost to selected nodes and 0 to others
        for node in self.nodes:
            if node in metered_nodes:
                node.cell_cost = random.uniform(0, 1)
            else:
                node.cell_cost = 0
