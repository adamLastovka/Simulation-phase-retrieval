import matplotlib.pyplot as plt

from simulation_module import *
import time
import cv2

if __name__ == "__main__":
    # Sim initialization
    wavelength = 1.032 * um
    computation_window = (15.9*mm, 15.9*mm)  # Spacial dimensions of computation window
    supersample_factor = 1
    sampling_size = int(1272*supersample_factor)  # sampling of computation window
    block_zero_order = False

    # Read mask + pre-process
    # mask = cv2.imread(r'C:\Users\lasto\OneDrive - University of Waterloo\Desktop\HiLase\lipss_classification\Phase Retreival\training_data\6_small_mask.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(r'C:\Users\lasto\OneDrive - University of Waterloo\Desktop\HiLase\lipss_classification\Phase Retreival\masks\pulsar-side\tophat_rect.bmp', cv2.IMREAD_GRAYSCALE)
    mask = mask[:, 4:-4] #Change masks to 1024x1072

    # Make Square
    mask = make_square(mask)

    # Resize while respecting SLM resolution (keep pixelation)
    if supersample_factor > 1:
        mask = supersample(mask, supersample_factor)

    start = time.time()

    sim = PropagationSimulation(wavelength, computation_window, sampling_size, mask, block_zero_order) # create simulation obj
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
