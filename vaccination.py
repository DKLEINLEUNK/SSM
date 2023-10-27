from main import SIRNetworkModel

# Configurate
random_SIR = SIRNetworkModel(topology='random', y=[1000, 0.01])
random_SIR.configure_SIR_model(beta=0.01, gamma=0.005, infected_nodes=5)

# Simulate
# random_SIR.run_simulation(1000)
random_SIR.run_multi_simulation(1000, 4)
random_SIR.plot_epidemic_spread(multi=True)

### Example Usage ###
# np.random.seed(42069) # for reproducibility
# NUM_NODES = 750
# P = 0.01
# random_network_SIR = SIRNetworkModel('random', [NUM_NODES, P])

# random_network_SIR.configure_SIR_model(beta=0.01, gamma=0.005, infected_nodes=5)

# # Plot network
# random_network_SIR.plot_network()

# # Run simulation
# random_network_SIR.run_simulation(1000)

# # Plot diffusion trend and prevalence
# random_network_SIR.plot_diffusion_trend(save=True)
# random_network_SIR.plot_epidemic_spread()
# # random_network_SIR.plot_diffusion_prevalence(save=True)

# # Plot degree distribution
# # random_network_SIR.plot_degree_distribution()
# random_network_SIR.plot_degree_distribution(expected=True, save=False)

# Get centralities
# random_network_SIR.get_centralities(verbose=True)