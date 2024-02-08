import hashlib
import random
import time
from typing import List

from ether.core import Node

class SymphonyOverlay:
    def __init__(self, nodes: List[Node], seed=None):
        self.nodes = nodes
        for node in self.nodes:
            node.symphony_id = self.generate_symphony_id(node, seed)

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

    def set_links(self):
        sorted_nodes = sorted(self.nodes, key=lambda x: x.symphony_id)
        for i, node in enumerate(sorted_nodes):
            num_nodes = len(sorted_nodes)
            node.successor_links = [sorted_nodes[(i + 1) % num_nodes], sorted_nodes[(i + 2) % num_nodes]]
            node.predecessor_links = [sorted_nodes[(i - 1) % num_nodes], sorted_nodes[(i - 2) % num_nodes]]
            # Additional logic for long_distance_links can be added here
