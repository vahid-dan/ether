import numpy as np
import srds

from ether.blocks.cells import IoTComputeBox, Cloudlet, FiberToExchange, MobileConnection
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
def calculate_total_cell_cost(path):
    total_cell_cost = 0
    incoming_traffic_cost = 1
    outgoing_traffic_cost = 1
    for i in range(len(path)):
        # Skip the incoming traffic cost for the first node
        if i > 0:
            if path[i].name.startswith('rpi4'):
                total_cell_cost += incoming_traffic_cost

        # Skip the outgoing cost for the last node
        if i < len(path) - 1:
            if path[i].name.startswith('rpi4'):
                total_cell_cost += outgoing_traffic_cost

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


def is_edge_node(node: Node) -> bool:
    edge_types = {'sbc', 'embai'}
    return node.labels.get('ether.edgerun.io/type') in edge_types
