import matplotlib.pyplot as plt
import srds
import random
import numpy as np

import ether.blocks.nodes as nodes
from ether.blocks.cells import IoTComputeBox, Cloudlet, FiberToExchange, MobileConnection
from ether.cell import GeoCell
from ether.core import Node, Link
from ether.topology import Topology
from ether.vis import draw_basic, print_symphony_structure
from ether.overlay import SymphonyOverlay

lognorm = srds.ParameterizedDistribution.lognorm


def node_name(obj):
    if isinstance(obj, Node):
        return obj.name
    elif isinstance(obj, Link):
        return f'link_{id(obj)}'
    else:
        return str(obj)


def generate_topology(num_neighborhoods,
                      num_nodes_per_neighborhood,
                      num_racks,
                      num_servers_per_rack,
                      node_type=nodes.rpi4,
                      density_params=(0.82, 2.02)):
    topology = Topology()

    def create_neighborhood(size):
        neighborhood_nodes = []
        for _ in range(size):
            node = IoTComputeBox(nodes=[node_type], backhaul=MobileConnection('internet_chix'))
            neighborhood_nodes.append(node)
        return neighborhood_nodes

    city = GeoCell(
        num_neighborhoods,
        nodes=create_neighborhood(num_nodes_per_neighborhood),
        density=lognorm(density_params))

    cloudlet = Cloudlet(
        num_servers_per_rack,
        num_racks,
        backhaul=FiberToExchange('internet_chix'))

    topology.add(city)
    topology.add(cloudlet)

    return topology


def main(num_neighborhoods=3,
         num_nodes_per_neighborhood=5,
         num_racks=2,
         num_servers_per_rack=4,
         node_type=nodes.rpi4,
         density_params=(0.82, 2.02)):
    topology = generate_topology(num_neighborhoods, num_nodes_per_neighborhood, num_racks, num_servers_per_rack, node_type, density_params)

    overlay_nodes = [Node(f"Node {i}") for i in range(len(topology.get_nodes()))]
    
    # Initialize the Symphony overlay with these nodes
    symphony_overlay = SymphonyOverlay(overlay_nodes, seed=SEED)
    symphony_overlay.set_links()  # This sets up the Symphony overlay links

    draw_basic(topology)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()  # display

    num_nodes = len(topology.get_nodes())
    print(f'Number of Nodes: {num_nodes}')

    print_symphony_structure(symphony_overlay)

if __name__ == '__main__':
    SEED = 42 # Use SEED in random functions
    random.seed(SEED)
    srds.seed(SEED)
    np.random.seed(SEED)
    num_neighborhoods = 3
    num_nodes_per_neighborhood = 5
    num_racks = 2
    num_servers_per_rack = 4
    node_type = nodes.rpi4
    density_params = (0.82, 2.02)
    main(num_neighborhoods, num_nodes_per_neighborhood, num_racks, num_servers_per_rack, node_type, density_params)
