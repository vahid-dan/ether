import networkx as nx

from ether.core import Node, Link
from ether.core import Node, Link
from ether.overlay import SymphonyOverlay


def draw_basic(topology):
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
    # Assuming SymphonyOverlay has a method or attribute to get all nodes with overlay info
    all_nodes = sorted(symphony_overlay.nodes, key=lambda node: node.symphony_id)
    
    for node in all_nodes:
        # Accessing the Symphony overlay information
        successors = [(successor.name, successor.symphony_id) for successor in node.successor_links]
        predecessors = [(predecessor.name, predecessor.symphony_id) for predecessor in node.predecessor_links]
        # long_distances = [(ld.name, ld.symphony_id) for ld in node.long_distance_links]
        
        print(f"{node.name} (ID: {node.symphony_id:.6f})")
        print(f"  -> Successors: {['%s (ID: %.6f)' % (name, id) for name, id in successors]}")
        print(f"  -> Predecessors: {['%s (ID: %.6f)' % (name, id) for name, id in predecessors]}")
        # print(f"  -> Long-Distances: {['%s (ID: %.6f)' % (name, id) for name, id in long_distances]}")
        print("-" * 50)
