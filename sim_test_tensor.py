from simulation_module_tensor import *
import cv2

if __name__ == "__main__":
    # Sim initialization
    wavelength = 1.032 * um
    computation_window = (15.9*mm, 15.9*mm)  # Spacial dimensions of computation window
    sampling_size = 1280  # sampling of computation window

    # Read mask + target
    mask_read = cv2.imread(r'C:\Users\lasto\OneDrive - University of Waterloo\Desktop\HiLase\lipss_classification\Phase Retreival\training_data\1_small_mask_gen.png', cv2.IMREAD_GRAYSCALE)

    sim = PropagationSimulation(wavelength, computation_window, sampling_size, mask_read)

    sim.propagate_forward()

    # Plots
    sim.draw_mask()

    sim.intensity.cut_resample
    sim.draw_image('intensity')

    intensity = sim.extract_intensity()
    plt.matshow(intensity.numpy())
    plt.show()

    counts, bins = np.histogram(intensity.numpy())
    plt.stairs(counts, bins)
    plt.show()
