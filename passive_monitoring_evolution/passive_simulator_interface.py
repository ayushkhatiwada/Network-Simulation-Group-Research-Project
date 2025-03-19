class PassiveMonitoringInterface:
    def __init__(self, ground_truth_network):
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
