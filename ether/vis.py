import networkx as nx
import matplotlib.pyplot as plt
import math
import os

from ether.core import Node, Link
from ether.core import Node, Link
from ether.overlay import SymphonyOverlay
from ether.topology import Topology


def draw_basic(topology: Topology):
    pos = nx.kamada_kawai_layout(topology)  # positions for all nodes

    # nodes

    hosts = [node for node in topology.nodes if isinstance(node, Node)]
    links = [node for node in topology.nodes if isinstance(node, Link)]
    switches = [node for node in topology.nodes if str(node).startswith('switch_')]

    nx.draw_networkx_nodes(topology, pos,
                           nodelist=hosts,
                           node_color='b',
                           node_size=300,
                           alpha=0.8)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=links,
                           node_color='g',
                           node_size=50,
                           alpha=0.9)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=switches,
                           node_color='y',
                           node_size=200,
                           alpha=0.8)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=[node for node in topology.nodes if
                                     isinstance(node, str) and node.startswith('internet')],
                           node_color='r',
                           node_size=800,
                           alpha=0.8)

    nx.draw_networkx_edges(topology, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(topology, pos, dict(zip(hosts, hosts)), font_size=10)
    nx.draw_networkx_labels(topology, pos, dict(zip(links, [l.tags['type'] for l in links])), font_size=8)
    # nx.draw_networkx_labels(topology, pos, dict(zip(links, links)), font_size=8)


def print_symphony_structure(symphony_overlay: SymphonyOverlay):
    all_nodes = sorted(symphony_overlay.nodes, key=lambda node: node.symphony_id)
    
    for node in all_nodes:
        # Accessing the Symphony overlay information
        successors = [(successor.name, successor.symphony_id) for successor in node.successor_links]
        predecessors = [(predecessor.name, predecessor.symphony_id) for predecessor in node.predecessor_links]
        long_distances = [(ld.name, ld.symphony_id) for ld in node.long_distance_links]
        
        print(f"{node.name} (ID: {node.symphony_id:.6f})")
        print(f"  -> Successors: {['%s (ID: %.6f)' % (name, id) for name, id in successors]}")
        print(f"  -> Predecessors: {['%s (ID: %.6f)' % (name, id) for name, id in predecessors]}")
        print(f"  -> Long-Distances: {['%s (ID: %.6f)' % (name, id) for name, id in long_distances]}")
        print("-" * 50)


def visualize_symphony_structure(topology: Topology):
    G = nx.DiGraph()

    # Add nodes to the graph and sort by Symphony ID
    nodes = sorted(topology.get_nodes(), key=lambda node: node.symphony_id)

    for node in nodes:
        G.add_node(node.name, id=node.symphony_id)

    # Add successor and long-distance links as edges
    for node in topology.get_nodes():
        for successor in node.successor_links:
            G.add_edge(node.name, successor.name, type='successor')
        for long_distance in node.long_distance_links:
            G.add_edge(node.name, long_distance.name, type='long_distance')

    # Calculate positions
    pos = {}
    num_nodes = len(nodes)

    for i, node in enumerate(G.nodes(data=True)):
        angle = (-2 * math.pi * i) / num_nodes + math.pi / 2  # Distribute nodes evenly on the ring
        pos[node[0]] = (math.cos(angle), math.sin(angle))

    # Create a copy of the original positions for labels
    pos_labels = pos.copy()

    # Adjust label positions based on node location
    label_offset = 0.1  # Adjust this value for label offset
    for node, (x, y) in pos.items():
        angle = math.atan2(y, x)
        
        if angle > -math.pi/2 and angle < math.pi/2:
            # Node is on the right side
            pos_labels[node] = (x + label_offset, y)
        else:
            # Node is on the left side
            pos_labels[node] = (x - label_offset, y)

    # Differentiate between successor and long-distance links
    edges_successor = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'successor']
    edges_long_distance = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'long_distance']

    plt.figure(figsize=(10, 10))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color="brown")

    # Draw the edges with different styles
    nx.draw_networkx_edges(G, pos, edgelist=edges_successor, edge_color='b', style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=edges_long_distance, edge_color='y', style='solid')

    # Draw node labels with adjusted positions
    nx.draw_networkx_labels(G, pos_labels, font_size=10)

    plt.title("Symphony Overlay Visualization")
    file_path = "results/symphony_overlay_visualization.png"
    plt.savefig(file_path)
    plt.show()


def visualize_topology(topology: Topology):
    draw_basic(topology)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    file_path = "results/topology_visualization.png"
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(file_path)
    plt.show()
