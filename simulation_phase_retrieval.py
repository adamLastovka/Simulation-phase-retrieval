from simulation_module import *
import matplotlib.pyplot as plt
import cv2


def create_square(square_width=50, sampling_size=1280, supersample_factor=1):
    """ 1 px = 12.5um"""
    target = np.zeros((sampling_size, sampling_size))
    square_width = int(square_width//2 * supersample_factor)

    target[sampling_size//2 - square_width:sampling_size//2 + square_width, sampling_size//2 - square_width:sampling_size//2 + square_width] = 1.0
    return target


def create_line(line_width=2, line_length=20, sampling_size=1280, supersample_factor=1):
    """ 1 px = 12.5um"""
    line_width = int(line_width // 2 * supersample_factor)
    line_length = int(line_length // 2 * supersample_factor)

    target = np.zeros((sampling_size, sampling_size))

    target[sampling_size//2 - line_length:sampling_size//2 + line_length, sampling_size//2 - line_width:sampling_size//2 + line_width] = 1.0
    return target

if __name__ == "__main__":
    '''
    This script generates a phase mask to achieve a specified target 
    intensity distribution by running the phase retrieval algorithm.
    '''

    num_iterations = 20  # iterations of phase retrieval

    # Sim Parameters
    wavelength = 1.032 * um
    computation_window = (15.9 * mm, 15.9 * mm)  # Spatial dimensions of computation window
    supersample_factor = 1
    sampling_size = int(1272 * supersample_factor)  # sampling of computation window
    block_zero_order = False
    log_flag = False
    output_window = (4*mm, 4*mm)  # Spatial area that will be plotted

    # Target intensity definition
    target_path = r'Images\moon_target.png'
    target_read = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    target_read = make_square(target_read)

    # initial random mask 0-210 (resolution too high initially if supersampling)
    mask_read = np.random.randint(0, 210, size=(sampling_size-int(248*supersample_factor), sampling_size))
    mask_read = make_square(mask_read)

    # Create simulation object
    sim = PropagationSimulation(wavelength, computation_window, sampling_size, mask_read, block_zero_order)

    # Start phase retrieval
    generated_mask, MSE_log, SSIM_log, mask_log, intensity_log = sim.retrieve_mask(target_read, num_iterations, log_flag)
    result_intensity = sim.extract_intensity(output_window)  # get intensity array

    ### Plot results ###
    sim.mask.draw('phase')
    draw_field(sim.image_field, 'intensity', output_window)
    draw_field(sim.image_field, 'intensity', output_window,log_scale=True)

    # Error metrics
    plt.figure()
    plt.plot(range(num_iterations),MSE_log)
    plt.title("MSE")
    plt.xlabel("Iteration")
    plt.xlim([1, 20])
    plt.xticks(ticks=range(1, num_iterations + 1,2), labels=[str(x) for x in range(1, num_iterations + 1,2)])
    plt.grid(which='both')

    plt.figure()
    plt.plot(range(1, num_iterations + 1),[np.log(x) for x in MSE_log])
    plt.title("log(MSE)")
    plt.xlabel("Iteration")
    plt.xlim([1, 20])
    plt.xticks(ticks=range(1, num_iterations + 1,2), labels=[str(x) for x in range(1, num_iterations + 1,2)])
    plt.grid(which='both')
    plt.show()

    plt.figure()
    plt.plot(range(1, num_iterations + 1),SSIM_log)
    plt.title("SSIM")
    plt.xlabel("Iteration")
    plt.xlim([1, 20])
    plt.xticks(ticks=range(1, num_iterations + 1,2), labels=[str(x) for x in range(1, num_iterations + 1,2)])
    plt.grid(which='both')
    plt.show()

    plt.figure()
    plt.plot(range(1,num_iterations+1),[np.log(1-x) for x in SSIM_log])
    plt.title("log(1-SSIM)")
    plt.xlabel("Iteration")
    plt.xlim([1,20])
    plt.xticks(ticks=range(1,num_iterations+1,2),labels=[str(x)for x in range(1,num_iterations+1,2)])
    plt.grid(which='both')
    plt.show()

    # Plot mask and intensity at every 10 iterations
    if log_flag:
        for i in range(len(intensity_log)):
            draw_field(intensity_log[i],'intensity', output_window,log_scale=False)
            draw_field(mask_log[i], 'phase', computation_window, log_scale=False)

    cv2.imwrite(r'PR_results\PR_mask_output.png', generated_mask)
    cv2.imwrite(r'PR_results\PR_intensity_output.png', result_intensity*255/np.max(result_intensity))
