import csv
import statistics
from ether.util import generate_random_source_destination_pairs, calculate_total_latency, calculate_total_cell_cost, calculate_total_true_cell_cost

class Evaluation:
    def __init__(self, overlay, topology, num_pairs, results_dir, filename, random_pairs):
        self.overlay = overlay
        self.topology = topology
        self.num_pairs = num_pairs
        self.results_dir = results_dir
        self.filename = filename
        self.random_pairs = random_pairs
        self.results = []
        self.total_latencies = []
        self.total_cell_costs = []

    def evaluate(self):
        self.calculate_metrics()
        self.write_results()
        self.write_statistics()

    def calculate_metrics(self):
        for i, (source_node, destination_node) in enumerate(self.random_pairs, start=1):
            symphony_path = self.overlay.find_symphony_path(source_node, destination_node)
            total_latency = calculate_total_latency(symphony_path, self.topology)
            total_cell_cost = calculate_total_true_cell_cost(symphony_path)
            self.total_latencies.append(total_latency)
            self.total_cell_costs.append(total_cell_cost)
            self.results.append({
                "ID": i,
                "Source": source_node.name,
                "Destination": destination_node.name,
                "Latency": total_latency,
                "Cell Cost": total_cell_cost
            })

    def write_results(self):
        with open(f'{self.results_dir}/{self.filename}.csv', mode='w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['ID', 'Source', 'Destination', 'Latency', 'Cell Cost'])
            for result in self.results:
                writer.writerow([result["ID"], result["Source"], result["Destination"], result["Latency"], result["Cell Cost"]])

    def write_statistics(self):
        stats_filename = f'{self.results_dir}/{self.filename}_stats.csv'
        with open(stats_filename, mode='w', newline='') as stats_file:
            writer = csv.writer(stats_file)
            writer.writerow(['Statistic', 'Value'])
            self._write_stat(writer, 'Latency Average', statistics.mean(self.total_latencies))
            self._write_stat(writer, 'Latency Median', statistics.median(self.total_latencies))
            self._write_stat(writer, 'Latency Minimum', min(self.total_latencies))
            self._write_stat(writer, 'Latency Maximum', max(self.total_latencies))
            self._write_stat(writer, 'Latency Standard Deviation', statistics.stdev(self.total_latencies))
            self._write_stat(writer, 'Cell Cost Average', statistics.mean(self.total_cell_costs))
            self._write_stat(writer, 'Cell Cost Median', statistics.median(self.total_cell_costs))
            self._write_stat(writer, 'Cell Cost Minimum', min(self.total_cell_costs))
            self._write_stat(writer, 'Cell Cost Maximum', max(self.total_cell_costs))
            self._write_stat(writer, 'Cell Cost Standard Deviation', statistics.stdev(self.total_cell_costs))
            total_network_cell_cost = self.overlay.calculate_total_network_cell_cost()
            self._write_stat(writer, 'Total Network Cell Cost', total_network_cell_cost)

    def _write_stat(self, writer, label, value):
        writer.writerow([label, value])
