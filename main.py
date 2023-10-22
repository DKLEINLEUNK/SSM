import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


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
positioning = nx.spring_layout(network)  # Position of the nodes

edge_colour = 'black' 
edge_width = 0.01

nx.draw(network, positioning, node_size=10, node_color='r', with_labels=False, edge_color=edge_colour, width=edge_width)
plt.title("Erdos-Renyi Network", fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()