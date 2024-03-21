import random
import numpy as np
import pandas as pd
import srds
import os
import csv
import statistics

import ether.blocks.nodes as nodes
from ether.core import Node
from ether.vis import visualize_symphony_structure, visualize_topology, print_symphony_structure
from ether.overlay import SymphonyOverlay
from examples.vivaldi.util import execute_vivaldi
from ether.util import generate_topology, generate_random_source_destination_pairs, generate_random_source_all_pairs, generate_all_to_all_pairs, print_all_node_properties
from ether.simulation import NetworkSimulation
from ether.evaluation import Evaluation


def main(target_selection_strategy='harmonic',
         decision_method='topsis',
         weights=np.array([1, 1]),
         is_benefit=np.array([False, False]),
         select_server_decision_method='topsis',
         select_server_weights=np.array([1, 1]),
         select_server_is_benefit=np.array([True, False]),
         num_neighborhoods=3,
         num_nodes_per_neighborhood=5,
         num_cloudlets=2,
         num_racks=2,
         num_servers_per_rack=4,
         workload_quota=100,
         num_cloud_racks=1,
         num_cloud_servers_per_rack=1,
         node_types_and_shares=[(nodes.rpi4, 95), (nodes.rpi3, 5)],
         density_params=(0.82, 2.02),
         metered_edge_nodes_percentage=50,
         num_pairs=10,
         results_dir="examples/results",
         cost_per_unit=1,
         num_iterations=30):

    print("#################### Step 0: Topology Initialization ####################")

    topology = generate_topology(num_neighborhoods,
                                 num_nodes_per_neighborhood,
                                 num_cloudlets,
                                 num_racks,
                                 num_servers_per_rack,
                                 workload_quota,
                                 num_cloud_racks,
                                 num_cloud_servers_per_rack,
                                 node_types_and_shares,
                                 density_params)

    # Update Vivaldi coordinates based on network interactions for all nodes
    execute_vivaldi(topology, node_filter=lambda n: isinstance(n, Node), min_executions=300)

    # visualize_topology(topology, results_dir)

    all_nodes = topology.get_nodes()
    num_all_nodes = len(all_nodes)
    print(f'Number of All Nodes: {num_all_nodes}')

    # Format topology parameters into a string for the filename
    topology_params_str = f"{target_selection_strategy}_{decision_method}_w{weights[0]}-{weights[1]}_nn{num_neighborhoods}_npn{num_nodes_per_neighborhood}_nc{num_cloudlets}_nr{num_racks}_nspr{num_servers_per_rack}_dp{density_params[0]}-{density_params[1]}_menp{metered_edge_nodes_percentage}"

    print("#################### Step 1: Symphony Initialization ####################")

    symphony_overlay = SymphonyOverlay(all_nodes, seed=SEED)

    symphony_overlay.assign_cell_costs(metered_edge_nodes_percentage) # cell_cost: True/False
    symphony_overlay.set_successor_links()
    symphony_overlay.set_long_distance_links(topology=topology,
                                             target_selection_strategy=target_selection_strategy,
                                             decision_method=decision_method,
                                             weights=weights,
                                             is_benefit=is_benefit)
    visualize_symphony_structure(topology, results_dir)

    if decision_method == 'topsis':
        pendant_nodes = symphony_overlay.discover_pendant_nodes()
        symphony_overlay.remove_links_from_pendant_nodes(pendant_nodes)
        symphony_overlay.set_successor_links()
        symphony_overlay.remove_overlapping_long_distance_links()
        symphony_overlay.set_long_distance_links(topology=topology,
                                             target_selection_strategy=target_selection_strategy,
                                             decision_method=decision_method,
                                             weights=weights,
                                             is_benefit=is_benefit)   
        symphony_overlay.set_bridge_links(topology=topology,
                                      weights=weights,
                                      is_benefit=is_benefit)
        print(f"Initial Total Network Cost: {symphony_overlay.calculate_total_network_cell_cost()}")
        visualize_symphony_structure(topology, results_dir)

    print("#################### Step 2: Evaluation Initialization ####################")

    random_pairs = generate_random_source_destination_pairs(all_nodes, num_pairs)
    source_all_pairs = generate_random_source_all_pairs(all_nodes)
    all_pairs = generate_all_to_all_pairs(all_nodes)

    print("#################### Step 3: Evaluation ####################")

    filename = f'{topology_params_str}_pairs{num_pairs}_seed{SEED}'

    for iteration in range(1, num_iterations + 1):
        print(f"#################### Iteration {iteration} ####################")

        # Application Traffic Simulation
        simulation = NetworkSimulation(symphony_overlay)
        simulation.simulate_application_traffic(topology=topology,
                                                select_server_decision_method=select_server_decision_method,
                                                select_server_weights=select_server_weights,
                                                select_server_is_benefit=select_server_is_benefit)
        simulation.print_traffic_matrix()
        simulation.calculate_and_update_node_costs(cost_per_unit)
        simulation.print_node_costs()

        # Evaluation
        iteration_filename = f'{filename}_iteration{iteration}'
        evaluation = Evaluation(symphony_overlay, topology, num_pairs, results_dir, iteration_filename, all_pairs)
        evaluation.evaluate()
        print(f"Total Network Cost: {symphony_overlay.calculate_total_network_cell_cost()}")

        visualize_symphony_structure(topology, results_dir)

        # Overlay Soft Reset and Reshape
        symphony_overlay.soft_reset()
        symphony_overlay.set_successor_links()
        symphony_overlay.set_long_distance_links(topology=topology,
                                                 target_selection_strategy=target_selection_strategy,
                                                 decision_method=decision_method,
                                                 weights=weights,
                                                 is_benefit=is_benefit)
        if decision_method == 'topsis':
            pendant_nodes = symphony_overlay.discover_pendant_nodes()
            symphony_overlay.remove_links_from_pendant_nodes(pendant_nodes)
            symphony_overlay.set_successor_links()
            symphony_overlay.remove_overlapping_long_distance_links()
            symphony_overlay.set_long_distance_links(topology=topology,
                                                     target_selection_strategy=target_selection_strategy,
                                                     decision_method=decision_method,
                                                     weights=weights,
                                                     is_benefit=is_benefit)   
            symphony_overlay.set_bridge_links(topology=topology,
                                              weights=weights,
                                              is_benefit=is_benefit)

    print("#################### Step 4: Sanity Check ####################")
    
    # print_all_node_properties(all_nodes)

    print(f"Randomness Check {random.random()}")


if __name__ == '__main__':
    num_neighborhoods = 4
    num_nodes_per_neighborhood = 6
    num_cloudlets = 2
    num_racks = 1
    num_servers_per_rack = 2
    workload_quota = 3000
    num_cloud_racks = 1
    num_cloud_servers_per_rack = 1
    node_types_and_shares = [
        (nodes.rpi4, 90),
        (nodes.rpi3, 10)
    ]
    density_params = (0.82, 2.02)
    metered_edge_nodes_percentage = 100
    num_pairs = 100
    results_dir = 'examples/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    target_selection_strategy = 'harmonic'
    decision_method = "topsis"
    weights = np.array([1, 1])
    is_benefit = np.array([False, False])
    select_server_decision_method = "topsis"
    select_server_weights = np.array([1, 1])
    select_server_is_benefit = np.array([False, True])
    cost_per_unit = 0.1
    num_iterations = 3
    SEED = 8
    random.seed(SEED)
    srds.seed(SEED)
    np.random.seed(SEED)
    main(target_selection_strategy,
         decision_method,
         weights,
         is_benefit,
         select_server_decision_method,
         select_server_weights,
         select_server_is_benefit,
         num_neighborhoods,
         num_nodes_per_neighborhood,
         num_cloudlets,
         num_racks,
         num_servers_per_rack,
         workload_quota,
         num_cloud_racks,
         num_cloud_servers_per_rack,
         node_types_and_shares,
         density_params,
         metered_edge_nodes_percentage,
         num_pairs,
         results_dir,
         cost_per_unit,
         num_iterations)
