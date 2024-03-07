import numpy as np
import srds
from random import choices

from ether.blocks.cells import IoTComputeBox, Cloudlet, Cloud, FiberToExchange, MobileConnection
from ether.cell import GeoCell
from ether.core import Node, Link
from ether.topology import Topology


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
                      workload_quota,
                      num_cloud_racks,
                      num_cloud_servers_per_rack,
                      node_types_and_shares,
                      density_params):
    topology = Topology()


    def create_neighborhoods(count):
        neighborhood_nodes = []
        for _ in range(count):
            # Determine the node type for each node based on the specified shares
            node_type = choices(
                population=[nt[0] for nt in node_types_and_shares],
                weights=[nt[1] for nt in node_types_and_shares],
                k=1
            )[0]
            node = IoTComputeBox(nodes=[node_type], backhaul=MobileConnection('internet_chix'))
            neighborhood_nodes.append(node)
        return neighborhood_nodes


    def create_cloudlets(count, workload_quota):
        cloudlets = []
        for i in range(count):
            location_id = str(i)
            cloudlet = Cloudlet(
                num_servers_per_rack,
                num_racks,
                backhaul=FiberToExchange('internet_chix'),
                location_id=location_id,
                workload_quota=workload_quota)
            cloudlets.append(cloudlet)
        return cloudlets    


    def create_cloud():
        cloud = Cloud(
            num_cloud_servers_per_rack,
            num_cloud_racks,
            backhaul=FiberToExchange('internet_chix'))
        return cloud


    city = GeoCell(
        num_neighborhoods,
        nodes=create_neighborhoods(num_nodes_per_neighborhood),
        density=lognorm(density_params))
    
    topology.add(city)

    for cloudlet in create_cloudlets(num_cloudlets, workload_quota):
        topology.add(cloudlet)

    cloud = create_cloud()
    topology.add(cloud)

    return topology


def harmonic_random_number(num_nodes):
    # Generate the harmonic series
    harmonic_series = np.array([1.0 / (i + 1) for i in range(1, num_nodes + 1)])
    
    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(harmonic_series) / np.sum(harmonic_series)
    
    # Generate a uniform random number
    random_num = np.random.uniform()
    
    # Find the index where this random number would fit in the CDF
    index = np.searchsorted(cdf, random_num)
    
    # Return the value corresponding to this index
    return index + 1  # Adding 1 because index starts from 0


def topsis(matrix, weights, is_benefit):
    # Ensure the matrix is not empty
    if matrix.size == 0:
        raise ValueError("Input matrix is empty")

    # Calculate the Euclidean norm of each column
    column_norms = np.sqrt((matrix ** 2).sum(axis=0))

    # Initialize the normalized matrix
    norm_matrix = np.zeros_like(matrix)

    # Normalize the matrix, setting columns with zero norm to zero
    nonzero_columns = column_norms != 0
    norm_matrix[:, nonzero_columns] = matrix[:, nonzero_columns] / column_norms[nonzero_columns]

    # Weighted normalized matrix
    weighted_matrix = norm_matrix * weights

    # Ensure weighted_matrix is not empty
    if weighted_matrix.size == 0:
        raise ValueError("Weighted matrix is empty")

    # Initialize ideal best and worst arrays
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    # Determine the ideal best and worst for each criterion
    for i in range(weighted_matrix.shape[1]):
        if is_benefit[i]:
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        else:
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    # Calculate the distances to the ideal best and ideal worst
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # Calculate the performance score, avoiding division by zero
    score = np.divide(dist_worst, dist_best + dist_worst, out=np.zeros_like(dist_worst), where=(dist_best + dist_worst) != 0)

    return score

    
# Function to calculate total latency for a path
def calculate_total_latency(path, topology):
    total_latency = 0
    for i in range(len(path) - 1):
        total_latency += topology.latency(path[i], path[i + 1], use_coordinates=False)
    return total_latency


# Function to calculate total cell_cost for a path
# def calculate_total_cell_cost(path):
#     total_cell_cost = 0

#     for node in path:
#         if hasattr(node, 'cell_cost'):
#             total_cell_cost += node.cell_cost

#     return total_cell_cost


# Function to calculate total cell_cost for a path
def calculate_total_cell_cost(path):
    if not path:  # Early return if path is empty
        return 0

    total_cell_cost = 0
    path_length = len(path)

    for i, node in enumerate(path):
        if hasattr(node, 'cell_cost'):
            # For the first and last node, add cell_cost once
            if i == 0 or i == path_length - 1:
                total_cell_cost += node.cell_cost
            # For inner nodes, add cell_cost twice
            else:
                total_cell_cost += node.cell_cost * 2

    return total_cell_cost


def print_cell_costs(nodes):
    for node in nodes:
        if is_edge_node(node):
            print(f"{node.name} is an edge node.")
            cell_cost = getattr(node, 'cell_cost', None)
            if cell_cost:
                print(f"cell_cost {cell_cost}")
        elif is_server_node(node):
            print(f"{node.name} is a server node.")


def print_location_ids(nodes):
    for node in nodes:
        if is_server_node(node):
            print(f"{node.name} is a server node.")
            location_id = getattr(node, 'location_id', None)
            if location_id:
                print(f"location_id {location_id}")
        elif is_edge_node(node):
            print(f"{node.name} is an edge node.")


def is_server_node(node: Node) -> bool:
    return node.labels.get('ether.edgerun.io/type') == 'server'


def is_cloud_server_node(node: Node) -> bool:
    return node.labels.get('ether.edgerun.io/type') == 'cloud_server'


def is_edge_node(node: Node) -> bool:
    edge_node_types = {'sbc', 'embai'}
    return node.labels.get('ether.edgerun.io/type') in edge_node_types


def is_power_node(node):
    power_node_models = {'server', 'vm'}
    return node.labels.get('ether.edgerun.io/model') in power_node_models


def is_regular_node(node):
    regular_node_models = {'rpi4', 'nvidia_jetson_nx', 'nvidia_jetson_tx2', 'nvidia_jetson_nano'}
    return node.labels.get('ether.edgerun.io/model') in regular_node_models


def is_constrained_node(node):
    constrained_node_models = {'rpi3b+'}
    return node.labels.get('ether.edgerun.io/model') in constrained_node_models
