import hashlib
import numpy as np
import random
import time
from typing import List
import math

from ether.core import Node
from ether.util import harmonic_random_number, topsis, is_edge_node, is_server_node
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


    def set_long_distance_links(self, topology: Topology, max_num_links: int = 2, link_selection_method: str = 'topsis', candidate_list_size_factor: int = 2, weights=np.array([1, 1]), is_benefit=np.array([False, False])):
        """
        Randomly select a node and attempt to assign a single link, repeating for a specified number of iterations.
        """
        sorted_nodes = sorted(self.nodes, key=lambda x: x.symphony_id)
        num_nodes = len(sorted_nodes)
        selection_size_factor = round(math.log(num_nodes))
        total_iterations = num_nodes * max_num_links * selection_size_factor

        for _ in range(total_iterations):
            # Randomly select a node
            node_index = np.random.randint(0, num_nodes)
            node = sorted_nodes[node_index]

            # Skip nodes that have reached the max number of links
            if len(node.long_distance_links) >= max_num_links:
                continue

            potential_targets = []
            found_targets = 0
            attempt_counter = 0
            max_attempts = num_nodes * 10  # Adjusted to limit attempts for each iteration

            # Try to find one potential target
            while found_targets < selection_size_factor and attempt_counter < max_attempts:
                attempt_counter += 1
                # Select a random target based on harmonic distribution
                index = harmonic_random_number(num_nodes) - 1
                potential_target = sorted_nodes[index]

                if (len(potential_target.long_distance_links) < max_num_links and
                    potential_target != node and
                    potential_target not in node.successor_links and
                    potential_target not in node.predecessor_links and
                    potential_target not in node.long_distance_links):
                    potential_targets.append(potential_target)
                    found_targets += 1
            print(f"potential_targets for {node} {potential_targets}")
            # Apply the TOPSIS method for link selection if targets are found
            if potential_targets and link_selection_method == 'topsis':
                criteria = []
                for target in potential_targets:
                    # Example criteria calculation
                    link_latency = topology.latency(node, target, use_coordinates=True)
                    link_cell_cost = target.cell_cost
                    criteria.append([link_latency, link_cell_cost])
                print(f"criteria {criteria}")
                criteria_matrix = np.array(criteria)
                scores = topsis(criteria_matrix, weights, is_benefit)
                print(f"scores {scores}")
                best_target_index = np.argmax(scores)
                best_target = potential_targets[best_target_index]

                # Establish a link with the best target
                node.long_distance_links.append(best_target)
                best_target.long_distance_links.append(node)  # Ensure bidirectional links
                print(f"Preferred Link established between {node} and {best_target}")

            elif link_selection_method == 'random' and potential_targets:
                random_target = potential_targets[0]

                # Establish links with randomly selected targets, respecting max_num_links
                node.long_distance_links.append(random_target)
                random_target.long_distance_links.append(node)  # Ensure bidirectional links
                print(f"Random link established between {node} and {random_target}")
            print(f"Links for {node} {node.long_distance_links}")

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
