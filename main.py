# from bokeh.io import output_notebook, show
import matplotlib.pyplot as plt
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.utils import multi_runs
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
# from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend as BokehDiffusionTrend
# from ndlib.viz.bokeh.MultiPlot import MultiPlot
# import ndlib.models.epidemics as ep
import numpy as np
from scipy.stats import binom


class SIRNetworkModel:
    """Class for generating SIR network models."""

    def __init__(self, topology: str, y: list):
        """Initializes the SIR network model.

        Parameters
        ----------
        `topology` : str
            The topology of the network model. 
            The available topologies are:
            - `small-world`: Watts–Strogatz graph.
            - `random`: Erdős–Rényi graph.
            - `scale-free`: Barabási–Albert graph.

        `y` : list
            The parameters for the network model, 
            specified as a list. 
            The required parameters per topology are:
            - small-world: `[num_nodes, k, p]`
            - random: `[num_nodes, p]`
            - scale-free: `[num_nodes, m]`
        """
        # Set up network
        self.topology = topology
        self.y = y
        self.num_nodes = y[0]

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

        # Initialize params
        self.model = None
        self.iters = None
        self.trends = None
        self.node_degrees_freq = None
        self.t = None
        self.X = []
        self.Y = []
        self.Z = []
        self.X_mean = []
        self.Y_mean = []
        self.Z_mean = []
        self.X_std = []
        self.Y_std = []
        self.Z_std = []

    def configure_SIR_model(self, beta: float, gamma: float, I=None, infected_nodes=None):
        """Configures the SIR model on networks.

        Parameters
        ----------
        `beta` : float
            The probability of infection (standard = 0.01).
        `gamma` : float
            The probability of recovery (standard = 0.005).
        `I` : float
            The fraction of initially infected nodes (standard = 0.05).
        `infected_nodes` : int
            The number of initially infected nodes, chosen randomly (default = None).
            If specified, `I` is ignored.
        """
        model = ep.SIRModel(self.network)

        # Configuration
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', beta)
        cfg.add_model_parameter('gamma', gamma)
        if infected_nodes is None:
            cfg.add_model_parameter('fraction_infected', I)
        else:
            nodes = np.random.choice(
                self.num_nodes, size=infected_nodes, replace=False)
            cfg.add_model_initial_configuration('Infected', nodes)

            print(f"{infected_nodes} initially infected nodes: {nodes}")

        model.set_initial_status(cfg)
        self.model = model

    ### Simulation ###
    def run_simulation(self, iterations: int):
        """Runs the simulation."""
        self.iters = self.model.iteration_bunch(iterations)
        self.trends = self.model.build_trends(self.iters)

        # Store the data properly for own plots
        self.t = np.arange(iterations)
        # 0 -> S, 1 -> I, 2 -> R
        self.X = self.trends[0]['trends']['node_count'][0]
        self.Y = self.trends[0]['trends']['node_count'][1]
        self.Z = self.trends[0]['trends']['node_count'][2]

    def run_multi_simulation(self, iterations: int, num_simulations: int):
        infected_nodes = [tuple(np.random.choice(
            self.num_nodes, size=5, replace=False))]*num_simulations
        self.trends = multi_runs(self.model, execution_number=num_simulations,
                                 iteration_number=iterations, infection_sets=infected_nodes, nprocesses=4)

        self.t = np.arange(iterations)

        for i in range(num_simulations):
            self.X.append(self.trends[i]['trends']['node_count'][0])
            self.Y.append(self.trends[i]['trends']['node_count'][1])
            self.Z.append(self.trends[i]['trends']['node_count'][2])

        self.X_mean, self.X_std = np.mean(
            self.X, axis=0), np.std(self.X, axis=0)
        self.Y_mean, self.Y_std = np.mean(
            self.Y, axis=0), np.std(self.Y, axis=0)
        self.Z_mean, self.Z_std = np.mean(
            self.Z, axis=0), np.std(self.Z, axis=0)

        ### DEBUG ###
        # # Verbose
        # print(f'\n\nLength of self.trends: {len(self.trends)}')
        # print('\n\nself.trends[0]:')
        # print(self.trends[0])
        # print('\n\nself.trends[1]:')
        # print(self.trends[1])
        # print('\n\nself.trends[2]:')
        # print(self.trends[2])
        # print(self.X_mean + (self.X_std))

        # # Pre-made plot (warning: may hurt eyes because so ugly)
        # viz = DiffusionTrend(self.model, self.trends)
        # viz.plot("diffusion.pdf", percentile=90)

    ### Visualization ###
    def plot_degree_distribution(self, expected=True, save=False):
        """Plots the degree distribution of the network.

        Parameters
        ----------
        `expected` : bool
            Whether to plot the expected degree distribution (default = True).
        `save` : bool
            Whether to save the plot (default = False).
        """
        if self.topology == 'random':
            self.plot_degree_distribution_random(expected, save)
        else:
            pass  # TODO add other topologies

    def plot_degree_distribution_random(self, expected=False, save=False):
        """Plots the degree distribution of a random network.

        Requires assumption of independent formation is met.
        """
        degree_freqs = nx.degree_histogram(
            self.network)  # frequencies of each degree value, k
        degrees = np.arange(len(degree_freqs))

        fig, ax = plt.subplots()
        # plt.figure(figsize=(12, 10)) # TODO add this back in
        ax.bar(degrees, degree_freqs)

        if expected:
            n = len(self.network)  # nodes
            m = self.network.size()  # edges
            # probability of edge formation, assumes independence
            p_k = 2 * m / (n * (n - 1))
            expected_freqs = [binom.pmf(k, n-1, p_k) * n for k in degrees]

            ax.plot(degrees, expected_freqs, 'bo-', label='Expected')

        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')

        plt.title(f'Degree distribution of {self.topology} network')
        plt.show()

        if save:
            plt.savefig('Plots/degree_distribution.png', dpi=400)

    def plot_degree_distribution_loglog(self, save=False):
        """Plots the degree distribution of the network on a log-log scale."""

    def plot_network(self, save=False):
        """Plots the network itself."""
        plt.figure(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(self.network)

        # Set node sizes & colors (based on degree)
        # TODO: add to init
        degrees = np.array([val for (node, val) in self.network.degree()])
        normalized_degrees = (degrees - min(degrees)) / \
            (max(degrees) - min(degrees))
        node_sizes = degrees * 15 + 10
        # fades from black to bright red
        node_colors = plt.cm.magma(normalized_degrees)

        # alpha = 0.9  # transparency

        nx.draw(
            self.network,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            # alpha=alpha,
            with_labels=False,
            # "Rich Grey" from https://colorcodes.io/gray/rich-gray-color-codes/
            edge_color="#1F2022",
            # note: this is the node border color (for some stupid reason...)
            edgecolors="black",
            linewidths=0.5,
            width=0.5,  # edge width
            # edge_alpha=alpha  TODO: add this back in for nx.draw_networkx_edges
        )

        plt.title(f"{self.topology} network".capitalize(),
                  fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.show()
        if save:
            plt.savefig('Plots/network.png', dpi=400)

    def plot_diffusion_trend(self, save=False):
        """Plots the diffusion trend and stores as `path`.png.

        As taken from `https://ndlib.readthedocs.io/en/latest/reference/viz/mpl/DiffusionTrend.html`:
        "The Diffusion Trend plot compares the trends of all the statuses allowed by the diffusive model tested.

        Each trend line describes the variation of the number of nodes for a given status iteration after iteration."
        """
        viz = DiffusionTrend(self.model, self.trends)
        if save:
            viz.plot('Plots/diffusion_trend.png')

    def plot_epidemic_spread(self, multi=False, save=False):
        """Alternative to `plot_diffusion_trend` because it was ugly.

        Parameters
        ----------
        `multi` : bool
            Whether run_multi_simulation() was used (default = False).
        """
        if multi:
            plt.figure(figsize=(10, 6))

            plt.plot(self.t, self.X_mean, label='Susceptible')
            plt.fill_between(self.t, 
                             (self.X_mean - 2*self.X_std), 
                             (self.X_mean + 2*self.X_std), 
                             alpha=0.2, label=r'$±2 SD$')

            plt.plot(self.t, self.Y_mean, label='Infected')
            plt.fill_between(self.t, 
                             (self.Y_mean - 2*self.Y_std), 
                             (self.Y_mean + 2*self.Y_std), 
                             alpha=0.2, label=r'$±2 SD$')

            plt.plot(self.t, self.Z_mean, label='Recovered')
            plt.fill_between(self.t, 
                             (self.Z_mean - 2*self.Z_std), 
                             (self.Z_mean + 2*self.Z_std), 
                             alpha=0.2, label=r'$±2 SD$')

            plt.xlabel('Time')
            plt.ylabel('Number of nodes')

            plt.title(
                f'SIR: Epidemic Spread on {self.topology.capitalize} Network')
            plt.legend()
            # plt.grid(True)
            plt.show()

        else:
            # print(f"{infected_nodes} initially infected nodes: {nodes}")
            plt.plot(self.t, self.X, label='Susceptible')
            plt.plot(self.t, self.Y, label='Infected')
            plt.plot(self.t, self.Z, label='Recovered')
            plt.xlabel('Time')
            plt.ylabel('Number of nodes')
            plt.title(
                f'SIR: Epidemic Spread on {self.topology.capitalize} Network')
            plt.legend()
            # plt.grid(True)
            plt.show()

        if save:
            plt.savefig('Plots/epidemic_spread.png', dpi=400)

    def plot_diffusion_prevalence(self, save=False):
        """Plots the diffusion prevalence and stores as `path`.png.

        As taken from `https://ndlib.readthedocs.io/en/latest/reference/viz/mpl/DiffusionPrevalence.html`:
        "The Diffusion Prevalence plot compares the delta-trends of all the statuses allowed by the diffusive model tested.

        Each trend line describes the delta of the number of nodes for a given status iteration after iteration."
        """
        viz = DiffusionPrevalence(self.model, self.trends)
        if save:
            viz.plot('Plots/diffusion_prevalence.png')

    ### Metrics ###
    def get_average_degree(self):
        """Returns the average degree of the network."""
        return np.mean([val for (node, val) in self.network.degree()])

    def get_centralities(self, verbose=False):
        """Returns the degree, betweenness, and closeness centralities of the network."""
        degree = nx.degree_centrality(self.network)
        betweenness = nx.betweenness_centrality(self.network)
        closeness = nx.closeness_centrality(self.network)
        # eigenvector = nx.eigenvector_centrality(G)
        # katz = nx.katz_centrality(G)
        # pagerank = nx.pagerank(G)
        # hits = nx.hits(G)

        if verbose:
            print(f"Degree centrality: {degree}")
            print(f"Betweenness centrality: {betweenness}")
            print(f"Closeness centrality: {closeness}")

        return degree, betweenness, closeness


### Example Usage ###
if __name__ == '__main__':
    np.random.seed(42069) # for reproducibility
    NUM_NODES = 750
    P = 0.01
    random_network_SIR = SIRNetworkModel('random', [NUM_NODES, P])

    random_network_SIR.configure_SIR_model(beta=0.01, gamma=0.005, infected_nodes=5)

    # Plot network
    random_network_SIR.plot_network()

    # Run simulation
    random_network_SIR.run_simulation(1000)

    # Plot diffusion trend and prevalence
    # random_network_SIR.plot_diffusion_trend(save=True)
    random_network_SIR.plot_epidemic_spread()
    # # random_network_SIR.plot_diffusion_prevalence(save=True)

    # # Plot degree distribution
    # # random_network_SIR.plot_degree_distribution()
    # random_network_SIR.plot_degree_distribution(expected=True, save=False)

    # Get centralities
    # random_network_SIR.get_centralities(verbose=True)
