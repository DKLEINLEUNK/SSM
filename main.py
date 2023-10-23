# import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


class SIRNetworkModel:
    """Class for generating SIR network models.

    Attributes
    ----------
    N : int
        The number of nodes in the network graph.

    y : list
        The parameters for the network model, specified as a list. The required parameters for each topology are as follows:
        - small-world: [num_nodes, k, p]
        - random: [num_nodes, p]
        - scale-free: [num_nodes, m]
    """

    def __init__(self, topology: str, y: list):
        
        self.topology = topology
        self.y = y

        if topology == "small-world":
            num_nodes, k, p = y[0], y[1], y[2]
            self.network = nx.watts_strogatz_graph(num_nodes, k, p)

        elif topology == "random":
            num_nodes, p = y[0], y[1]
            self.network = nx.gnp_random_graph(num_nodes, p)

        elif topology == "scale-free":
            num_nodes, m = y[0], y[1]
            self.network = nx.barabasi_albert_graph(num_nodes, m)

        else:
            raise ValueError("Invalid topology.")
        
    def configure_SIR_model(self, beta: float, gamma: float, I: float):
        """
        Configures the SIR model on networks.

        Parameters
        ----------
        beta : float
            The probability of infection (default = 0.01).
        gamma : float
            The probability of recovery (default = 0.005).
        I : float
            The fraction of initially infected nodes (default = 0.05).
        """
        model = ep.SIRModel(self.network)

        # Configuration
        cfg = mc.Configuration()
        cfg.add_model_parameter("beta", beta)
        cfg.add_model_parameter("gamma", gamma)
        cfg.add_model_parameter("fraction_infected", I)

        model.set_initial_status(cfg)


def sir_random_network(self, num_nodes=1000, p=0.1, beta=0.01, gamma=0.005, i=0.05):
    """
    Generates a random graph and runs an SIR model on it.

    Parameters
    ----------
    num_nodes : int, optional
        The number of nodes in the network graph (default is 1000).
    p : float, optional
        The probability of an edge between any two nodes in the network graph (default is 0.1).
    beta : float, optional
        The probability of infection (default is 0.01).
    gamma : float, optional
        The probability of recovery (default is 0.005).
    i : float, optional
        The fraction of initially infected nodes (default is 0.05).

    Returns
    -------
    network : networkx.Graph
        The generated network graph.
    model : ndlib.models.epidemics.SIRModel
        The SIR model object.
    """
    G = nx.gnp_random_graph(num_nodes, p)
    model = ep.SIRModel(G)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", beta)  # probability of infection
    cfg.add_model_parameter("gamma", gamma)  # probability of recovery
    cfg.add_model_parameter("fraction_infected", i)

    model.set_initial_status(cfg)

    return G, model

def run_simulation(self, model, iters):
    iterations = model.iteration_bunch(iters)
    trends = model.build_trends(iterations)

    return iterations, trends


# # Example use:
# network, model = SIR_random_network()
# iterations, trends = simulation(model, 200)
# plt.figure(figsize=(12, 10))
# positioning = nx.spring_layout(network)
# EDGE_COLOR = "black"
# EDGE_WIDTH = 0.01
# nx.draw(
#     network,
#     positioning,
#     node_size=10,
#     node_color="r",
#     with_labels=False,
#     edge_color=EDGE_COLOR,
#     width=EDGE_WIDTH,
# )
# plt.title("Erdos-Renyi Network", fontsize=16, fontweight="bold")
# plt.axis("off")
# plt.show()
