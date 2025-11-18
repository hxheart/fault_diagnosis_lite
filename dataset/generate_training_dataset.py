import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

import numpy as np

from semantics import FactBaseSemanticsDataset  # 1) FRAMEWORK:  how to create datasets
from bgp_semantics import BgpSemantics          # 3) CORE LOGIC: actual BGP/OSPF simulation
from nutils import choose_random

class ConfiguredBgpSemantics:                   # 2) CONFIGURATION: what parameters to use
    def __init__(self):
        self.s = BgpSemantics(labeled_networks=False) # BgpSemantics does the protocol simulation. Explore this to find out how BGP/OSPF simulation works

    def sample(self, seed):
        s = np.random.RandomState(seed=seed) # randomly generate some network parameters, such as number of nodes, number of ASes, ...

        # network parameters
        real_world_topology = False                         # np.random.random() < 0.2 # {True:0.3, False:0.7}
        num_networks = choose_random(list(range(4,8)), s)   # Number of networks
        num_gateway_nodes = 3                               # Number of gateway nodes
        num_nodes = choose_random(range(16,24), s)          # Total number of nodes

        sample_config_overrides = {
            "fwd":              {"n": choose_random([8, 10, 12], s)},   # num paths of fwd predicates
            "reachable":        {"n": choose_random([4, 5, 6, 7], s)},  # num reachable predicates
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},     # num trafficIsolation predicates
        }

        seed = s.randint(0,1024*1024*1024)

        # Here, this is using BgpSematics.sample()
        return self.s.sample(num_nodes=num_nodes,
                             real_world_topology=real_world_topology,
                             num_networks=num_networks,
                             predicate_semantics_sample_config_overrides=sample_config_overrides,
                             seed=seed,
                             NUM_GATEWAY_NODES=num_gateway_nodes)

if __name__ == "__main__":
    dataset = FactBaseSemanticsDataset(  #
        ConfiguredBgpSemantics(),        # here will create a BgpSemantics instance: "self.s = BgpSemantics(labeled_networks=False)", which is not empty;
        "bgp-ospf-dataset-sub",     # go to "root", just creating a path/folder;
        num_samples=10*1024,
        tmp_directory="tmp-bgp-dataset"
    )
    # print(len(dataset))
    # print('\n The first sample is:', dataset[0])
    # print('\n', dataset[0].keys())

    data = dataset[0]
    print('\n ===> data is:\n', data) # PyTorch Geometric (PyG) Data object: torch_geometric.data.Data
    print('\n ===> data.x is:\n', data.x)           # x represents: node feature matrix -- semantic information is encoded here!
    print('\n ===> data.x[0] is:\n', data.x[0])
    print('\n ===> data.edge_index is:\n', data.edge_index)  # Edge connectivity matrix
    print('\n ===> data.edge_attr is:\n', data.edge_attr)   # Edge attribute matrix

    # View all properties of a data object
    print("\n 1) ===> Data attributes:\n", dir(data))
    print("\n 2) ===> Data keys:\n", data.keys())

    # Accessing the underlying semantic objects
    semantics_visualise = ConfiguredBgpSemantics()
    sample_visualise = semantics_visualise.sample(seed=42)
    print("\n 3) ===> Sample from semantics:\n", sample_visualise)
    print("\n 4) ===> Type of sample:\n", type(sample_visualise))

# configuration facts: router, network, external, connected, ibgp, ebgp, bgp_route
# specification facts: fwd, reachable, trafficIsolation


