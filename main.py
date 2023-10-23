# import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


class SIRNetworkModel:
    """Class for generating SIR network models.

    Attributes
    ----------
    topology : str
        The topology of the network model. The available topologies are:
        - `small-world`: Watts–Strogatz graph.
        - `random`: Erdős–Rényi graph.
        - `scale-free`: Barabási–Albert graph.

    y : list
        The parameters for the network model, specified as a list. The required parameters for each topology are as follows:
        - small-world: `[num_nodes, k, p]`
        - random: `[num_nodes, p]`
        - scale-free: `[num_nodes, m]`
    """

    def __init__(self, topology: str, y: list):
        """Initializes the SIR network model.

        Parameters
        ----------
        topology : str
            The topology of the network model. The available topologies are:
            - `small-world`: Watts–Strogatz graph.
            - `random`: Erdős–Rényi graph.
            - `scale-free`: Barabási–Albert graph.

        y : list
            The parameters for the network model, specified as a list. The required parameters for each topology are as follows:
            - small-world: `[num_nodes, k, p]`
            - random: `[num_nodes, p]`
            - scale-free: `[num_nodes, m]`
        """
        # Set the network topology
        self.topology = topology
        self.y = y

        if topology == 'small-world':
            num_nodes, k, p = y[0], y[1], y[2]
            self.network = nx.watts_strogatz_graph(num_nodes, k, p)

        elif topology == 'random':
            num_nodes, p = y[0], y[1]
            self.network = nx.gnp_random_graph(num_nodes, p)

        elif topology == 'scale-free':
            num_nodes, m = y[0], y[1]
            self.network = nx.barabasi_albert_graph(num_nodes, m)

        else:
            raise ValueError('Invalid topology.')

        # Set the network to undirected
        self.model = None
        self.iters = None
        self.trends = None

    def configure_SIR_model(self, beta: float, gamma: float, I: float):
        """Configures the SIR model on networks.

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
        cfg.add_model_parameter('beta', beta)
        cfg.add_model_parameter('gamma', gamma)
        cfg.add_model_parameter('fraction_infected', I)

        model.set_initial_status(cfg)

        self.model = model

    def run_simulation(self, iterations: int):
        """Runs the simulation.

        Parameters
        ----------
        iterations : int
            The number of iterations to run the simulation for.
        """
        self.iters = self.model.iteration_bunch(iterations)
        self.trends = self.model.build_trends(self.iters)


### Example Usage ###
num_nodes = 1000
p = 0.01
SIR_network_model = SIRNetworkModel('random', [num_nodes, p])
SIR_network_model.configure_SIR_model(beta=0.01, gamma=0.005, I=0.05)
SIR_network_model.run_simulation(200)

# Fetch the network and trends
network = SIR_network_model.network
iterations, trends = SIR_network_model.iters, SIR_network_model.trends

# Plot the trends
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
positioning = nx.spring_layout(network)
EDGE_COLOR = "black"
EDGE_WIDTH = 0.01
nx.draw(
    network,
    positioning,
    node_size=10,
    node_color="r",
    with_labels=False,
    edge_color=EDGE_COLOR,
    width=EDGE_WIDTH,
)
plt.title("Erdos-Renyi Network", fontsize=16, fontweight="bold")
plt.axis("off")
plt.show()
