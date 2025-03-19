from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class PassiveMonitoringSimulator:
    def __init__(self, paths="1"):
        self.network = GroundTruthNetwork(paths)
        self.switches = {
            ground_truth_network.SOURCE: ground_truth_network.source_switch,
            ground_truth_network.DESTINATION: ground_truth_network.destination_switch
        }
    
    def attach_sketch(self, node_id, sketch):
        # Node id is the value 1 or 2 in this case for the simplest one edge two nodes graph
        if node_id in self.switches:
            self.switches[node_id].add_sketch(sketch)
            print(f"Attached {sketch.__class__.__name__} to switch {node_id}.")
        else:
            raise ValueError(f"No switch found for node id {node_id}.")

    def compare_distribution_parameters(self, pred_mean: float, pred_std: float) -> float:
        """
        Compares the predicted delay distribution parameters with the actual network parameters using KL divergence.
        Aim for a KL divergence of <= 0.05
        """
        params = self.network.get_distribution_parameters(self.network.SOURCE, self.network.DESTINATION)
        actual_mean = params["mean"]
        actual_std = params["std"]

        kl_div = math.log(pred_std / actual_std) + ((actual_std**2 + (actual_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        
        # Aim for a KL divergence of <= 0.05
        if kl_div <= 0.05:
            print(f"KL divergence: {kl_div:.4f} ✅")
        else:
            print(f"KL divergence: {kl_div:.4f} ❌")
        return kl_div
