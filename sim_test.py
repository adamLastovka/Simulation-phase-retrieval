import matplotlib.pyplot as plt
from simulation_module import *
import time
import cv2

if __name__ == "__main__":
    '''
    This script simulates the laser propagation through a specified mask and plots the output intensity distribution.
    '''

    # Sim initialization
    wavelength = 1.032 * um
    computation_window = (15.9*mm, 15.9*mm)  # Spacial dimensions of computation window
    supersample_factor = 1
    sampling_size = int(1272*supersample_factor)  # sampling of computation window
    block_zero_order = False  # Flag to enable zero order blocking

    # Read mask + pre-process
    mask_path = r'Images\tophat_rect.bmp'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask using CV2
    mask = mask[:, 4:-4]  # Change masks to 1024x1072

    # Make Square
    mask = make_square(mask)

    # Resize while respecting SLM resolution (keep pixelation)
    if supersample_factor > 1:
        mask = supersample(mask, supersample_factor)

    start = time.time()

    # create simulation object and run sim
    sim = PropagationSimulation(wavelength, computation_window, sampling_size, mask, block_zero_order)
    sim.propagate_forward()

    end = time.time()
    print(f"Simulation Time: {end-start}s")

    # Plots
    sim.mask.draw('phase')
    plot_window = (0.4 * mm, 0.4 * mm)
    draw_field(sim.image_field,'intensity', plot_window)
    draw_field(sim.image_field,'intensity', plot_window, log_scale=True)

    # intensity = sim.extract_intensity()
    # plt.matshow(np.log(intensity))
    # plt.show()
