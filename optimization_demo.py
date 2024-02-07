import matplotlib.pyplot as plt
import torch

from optimization_module import *
from simulation_module_tensor import *
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Set the backend TkAgg

if __name__ == "__main__":
    # Sim initialization
    wavelength = 1.032 * um
    computation_window = (15.9*mm, 15.9*mm)  # Spacial dimensions of computation window
    sampling_size = 1280  # sampling of computation window

    # Read mask + target
    mask_read = cv2.imread(r'C:\Users\lasto\OneDrive - University of Waterloo\Desktop\HiLase\lipss_classification\Phase Retreival\training_data\square_mask_150px_gen.png', cv2.IMREAD_GRAYSCALE) # initial mask

    target_read = cv2.imread(r'C:\Users\lasto\OneDrive - University of Waterloo\Desktop\HiLase\lipss_classification\Phase Retreival\training_data\square_target_150px.png', cv2.IMREAD_GRAYSCALE)  # TODO: define target spatial size (currently assuming 15.9 wide,12.8 tall)
    target_intensity = (target_read / 255.0)  # scale to 0-1

    sim = PropagationSimulation(wavelength, computation_window, sampling_size, mask_read)

    import os
    import sys
    os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
    sys.path.append(r'C:\Program Files\Graphviz\bin')

    # Optimization params
    config = {
        'lr': 500,
        'momentum': 0.9,
        'num_iterations': 50,
    }

    # Run optimization
    mask_read = slm2phase(mask_read)
    net, train_losses = train_net(sim, mask_read, target_intensity, config)

    # Plot results
    plot_training_losses(train_losses)

    cv2.imwrite("optimized_mask_test.png", phase2slm(np.angle(sim.mask[128:1152,:].detach())))
