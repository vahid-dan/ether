import math
import srds
import random
import numpy as np

import ether.blocks.nodes as nodes
from ether.blocks.cells import IoTComputeBox, Cloudlet, FiberToExchange, MobileConnection
from ether.cell import GeoCell
from ether.core import Node, Link
from ether.topology import Topology
from ether.vis import visualize_symphony_structure, visualize_topology
from ether.overlay import SymphonyOverlay
from examples.vivaldi.util import execute_vivaldi
from ether.util import print_location_ids


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
                      num_cloudlets,
                      num_racks,
                      num_servers_per_rack,
                      node_type,
                      density_params):
    topology = Topology()


    def create_neighborhoods(count):
        neighborhood_nodes = []
        for _ in range(count):
            node = IoTComputeBox(nodes=[node_type], backhaul=MobileConnection('internet_chix'))
            neighborhood_nodes.append(node)
        return neighborhood_nodes


    def create_cloudlets(count):
        cloudlets = []
        for i in range(count):
            location_id = str(i)
            cloudlet = Cloudlet(
                num_servers_per_rack,
                num_racks,
                backhaul=FiberToExchange('internet_chix'),
                location_id=location_id)
            cloudlets.append(cloudlet)
        return cloudlets    


    city = GeoCell(
        num_neighborhoods,
        nodes=create_neighborhoods(num_nodes_per_neighborhood),
        density=lognorm(density_params))
    
    topology.add(city)
    for cloudlet in create_cloudlets(num_cloudlets):
        topology.add(cloudlet)
    return topology


def main(num_neighborhoods=3,
         num_nodes_per_neighborhood=5,
         num_cloudlets=2,
         num_racks=2,
         num_servers_per_rack=4,
         node_type=nodes.rpi4,
         density_params=(0.82, 2.02),
         metered_edge_nodes_percentage=50):
    topology = generate_topology(num_neighborhoods,
                                 num_nodes_per_neighborhood,
                                 num_cloudlets,
                                 num_racks,
                                 num_servers_per_rack,
                                 node_type,
                                 density_params)

    # Update Vivaldi coordinates based on network interactions for all nodes
    execute_vivaldi(topology, node_filter=lambda n: isinstance(n, Node), min_executions=300)

    # Use the nodes from the topology as overlay nodes
    overlay_nodes = topology.get_nodes()
    
    num_nodes = len(overlay_nodes) # Number of nodes
    max_num_links = round(math.log(num_nodes)) # Max number of long-distance links per node

    # Initialize the Symphony overlay with these nodes
    symphony_overlay = SymphonyOverlay(overlay_nodes, seed=SEED)
    symphony_overlay.assign_cell_cost(metered_edge_nodes_percentage)
    symphony_overlay.set_successor_links()    
    symphony_overlay.set_long_distance_links(topology=topology,
                                             max_num_links=max_num_links,
                                             link_selection_method='random',
                                             candidate_list_size_factor=2,
                                             weights=[1, 1],
                                             is_benefit=[False, False])

    print_location_ids(overlay_nodes)

    visualize_topology(topology)

    print(f'Number of Nodes: {num_nodes}')

    # print_symphony_structure(symphony_overlay)

    visualize_symphony_structure(topology)

if __name__ == '__main__':
    SEED = 42 # Use SEED in random functions
    random.seed(SEED)
    srds.seed(SEED)
    np.random.seed(SEED)
    num_neighborhoods = 3
    num_nodes_per_neighborhood = 5
    num_cloudlets = 2
    num_racks = 2
    num_servers_per_rack = 4
    node_type = nodes.rpi4
    density_params = (0.82, 2.02)
    metered_edge_nodes_percentage = 50
    main(num_neighborhoods,
         num_nodes_per_neighborhood,
         num_cloudlets,
         num_racks,
         num_servers_per_rack,
         node_type,
         density_params,
         metered_edge_nodes_percentage)
