import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

def generate_network(type, N=2000, p=0.1, m=2, k=4):
    """Generates a network of specified type with coresponding parameters.

    Parameters
    ----------
    type : NetworkX graph
        Available options include: 'small-world', 'random', 
        & 'scale-free'.
    N : int
        Total number of nodes.
    p : float
        Probability of linking (or rewiring) two nodes.
    m : int
        Number of edges to attach from a new node to existing nodes
    k : int
        Each node is joined with `k` nearest neighbors.
    """
    if type == 'small-world': return nx.watts_strogatz_graph(N, k, p)
    elif type == 'random': return nx.erdos_renyi_graph(N, p)
    elif type == 'scale-free': return nx.barabasi_albert_graph(N, m)


def SIR_random_network(N=1000, p=0.1, beta=0.01, gamma=0.005, I=0.05):
    network = nx.erdos_renyi_graph(N, p)
    model = ep.SIRModel(network)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)  # probability of infection
    cfg.add_model_parameter('gamma', gamma)  # probability of recovery
    cfg.add_model_parameter("fraction_infected", I)

    model.set_initial_status(cfg)

    return network, model


def simulation(model, iters):
    iterations = model.iteration_bunch(iters)
    trends = model.build_trends(iterations)

    return iterations, trends


# Example use:
import matplotlib.pyplot as plt

network, model = SIR_random_network()
iterations, trends = simulation(model, 200)
plt.figure(figsize=(12, 10))
positioning = nx.spring_layout(network)
edge_colour = 'black' 
edge_width = 0.01
nx.draw(network, positioning, node_size=10, node_color='r', with_labels=False, edge_color=edge_colour, width=edge_width)
plt.title("Erdos-Renyi Network", fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()