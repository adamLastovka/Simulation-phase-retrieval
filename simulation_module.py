import diffractio
from diffractio import plt, np, mm, degrees, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from matplotlib import cm
import cv2
import copy

diffractio.config.CONF_DRAWING['color_intensity'] = cm.jet
def phase2complex(mask_phase):
    return np.exp(1j * mask_phase)

def slm2phase(mask_slm):
    return (mask_slm / 210.0) * (2 * np.pi) - np.pi

def phase2slm(mask_phase):
    return (mask_phase + np.pi) / (2 * np.pi) * 210.0

def calculate_SSIM(img1, img2):
    """
    Calculates structural similarity index of two images
    """
    c1 = (0.01 * 255) ** 2  # Constants for stability
    c2 = (0.03 * 255) ** 2

    # Compute means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Compute variances
    var1 = np.var(img1)
    var2 = np.var(img2)

    # Compute covariance
    cov = np.cov(img1.flatten(), img2.flatten())[0, 1]

    # Compute SSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)

    ssim = numerator / denominator

    return ssim


def calculate_CC(image1, image2):
    """
    Calculates correlation coefficient of two images
    """
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same shape")

    cov = np.cov(image1, image2)[0, 1]

    # Compute variances
    var_f1 = np.var(image1)
    var_f2 = np.var(image2)

    # Compute correlation coefficient
    correlation_coefficient = cov / (np.sqrt(var_f1 * var_f2))

    return correlation_coefficient


def calculate_MSE(image1, image2):
    """ Calculates MSE"""
    return np.mean(((image1 - image2) ** 2))


def supersample(raw_image, factor):
    """ Resamples image by specified integer factor while maintaining pixelation"""
    img_y, img_x = raw_image.shape
    resampled_image = np.zeros((img_y * factor, img_x * factor), dtype=np.float64)
    for y in range(img_y):
        for x in range(img_x):
            value = raw_image[y, x]
            resampled_image[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor] = value
    return resampled_image


def make_square(raw_image):
    """ Zero-pads image in smaller-dimension to make image square"""
    img_y, img_x = raw_image.shape
    if img_y > img_x:
        x_pad = int((img_y - img_x) / 2)
        square_image = np.pad(raw_image, ((0, 0), (x_pad, x_pad)), constant_values=0)
    elif img_x > img_y:
        y_pad = int((img_x - img_y) / 2)
        square_image = np.pad(raw_image, ((y_pad, y_pad), (0, 0)), constant_values=0)
    else:
        square_image = raw_image

    return square_image


def draw_field(field,type,spatial_window=None,log_scale=False):
    """Wrapper for drawing and resampling functions for ease of use"""
    if spatial_window is None:
        x_lim = None
        y_lim = None
    else:
        x_lim = (-spatial_window[0],spatial_window[0])
        y_lim = (-spatial_window[1], spatial_window[1])

    extracted_field = field.cut_resample(x_limits=x_lim, y_limits=y_lim, new_field=True, interp_kind=(3, 1))

    if type == "phase":
        extracted_field.draw('phase', logarithm=log_scale)
    elif type == "intensity":
        extracted_field.draw('intensity', logarithm=log_scale)
    else:
        print("Incorrect field type")
    plt.show()

def pixelate(image, factor):
    """pixelates image while maintaining dimensions"""
    img_y, img_x = image.shape
    pixel_image = cv2.resize(image,dsize=(img_y//factor,img_x//factor),interpolation=cv2.INTER_LINEAR)
    pixel_image = supersample(pixel_image, factor)
    return pixel_image

class PropagationSimulation:
    def __init__(self, wavelength, computation_window, sampling_size, mask_read=None, block_zero_order=False):
        """
        Initialize simulation object
        :param wavelength: Wavelength of laser (um)
        :param computation_window: Spatial size of computation window (y,x) (um). Should be square to avoid distortions
        :param sampling_size: Number of samples of computation window (int). Best performance with 2^k samples (reference?)
        :param mask_read: SLM phase mask (grayscale np.array, must match sampling size)
        """
        supersample_factor = sampling_size // 1280

        # Spatial dimensions
        x_size = computation_window[1]
        y_size = computation_window[0]

        # Vectors of sample points (uniform sampling distribution)
        x_vect = np.linspace(-x_size / 2, x_size / 2, sampling_size+1)[:-1]+x_size/(2*sampling_size)
        y_vect = np.linspace(-y_size / 2, y_size / 2, sampling_size+1)[:-1]+y_size/(2*sampling_size)

        # Source intensity
        u0 = Scalar_source_XY(x=x_vect, y=y_vect, wavelength=wavelength)
        u0.gauss_beam((0, 0), (6 * mm, 6 * mm), (0, 0))

        # SLM MASK Embedding
        if mask_read is None:  # random mask if none specified
            mask_read = np.random.randint(low=0, high=211, size=(sampling_size, sampling_size))
        assert mask_read.shape == (sampling_size, sampling_size), (f"Mask shape ({mask_read.shape}) does not match sampling size "
                                                                   f"({sampling_size, sampling_size})")

        self.mask = Scalar_mask_XY(x=x_vect, y=y_vect, wavelength=wavelength)  # create mask scalar field

        mask_read = slm2phase(mask_read)  # convert SLM to phase mask (scaling)
        self.mask.u = phase2complex(mask_read)  # set mask property

        # Lens masks
        lens_1 = Scalar_mask_XY(x=x_vect, y=y_vect, wavelength=wavelength)
        lens_1.lens_spherical(r0=(0, 0), focal=700 * mm, radius=30 * mm, refraction_index=1.5)

        lens_2 = Scalar_mask_XY(x=x_vect, y=y_vect, wavelength=wavelength)
        lens_2.lens_spherical(r0=(0, 0), focal=160 * mm, radius=30 * mm, refraction_index=1.5)

        # Backwards lens masks
        lens_1b = lens_1.conjugate(new_field=True)
        lens_2b = lens_2.conjugate(new_field=True)

        # Zero-order blocking
        ZO_width = 0.5 * mm // 2
        vertices = np.array([(-ZO_width, -ZO_width), (-ZO_width, ZO_width), (ZO_width, ZO_width),
                             (ZO_width, -ZO_width)])
        ZO_mask = Scalar_mask_XY(x_vect, y_vect, wavelength)
        ZO_mask.polygon(vertices)
        ZO_mask.inverse_amplitude()

        # Assign properties
        self.wavelength = wavelength
        self.sampling_size = sampling_size
        self.supersample_factor = supersample_factor
        self.x_size = x_size
        self.y_size = y_size
        self.x_vect = x_vect
        self.y_vect = y_vect
        self.input_field = u0
        self.lens_1 = lens_1
        self.lens_2 = lens_2
        self.lens_1b = lens_1b
        self.lens_2b = lens_2b
        self.ZO_mask = ZO_mask
        self.block_zero_order = block_zero_order
        self.image_field = []

    def propagate_forward(self):
        """ Propagates input field through mask and optics and yields output at the image field"""
        propagation_field = self.input_field * self.mask  # SLM

        # propagation_field.RS(z=100*mm, verbose=True, new_field=False) # has little effect and slows down sim
        propagation_field = propagation_field * self.lens_1

        propagation_field.RS(z=700 * mm, verbose=False, new_field=False) # to focus

        if self.block_zero_order:
            propagation_field = propagation_field * self.ZO_mask

        propagation_field.RS(z=350 * mm, verbose=False, new_field=False) # to f-Theta

        propagation_field = propagation_field * self.lens_2  # F-Theta

        propagation_field.RS(z=285 * mm, verbose=False, new_field=False) # To target

        self.image_field = propagation_field

    def propagate_backwards(self):
        """ Propagates the image field backwards through the optical system and yields field at the SLM"""
        propagation_field = self.image_field.RS(z=-285 * mm, verbose=False) # surface to f-Theta
        propagation_field = propagation_field * self.lens_2b # F-theta
        propagation_field.RS(z=-1050 * mm, verbose=False, new_field=False) # F-Theta to lens
        propagation_field = propagation_field * self.lens_1b # Lens
        # propagation_field.RS(z=-100 * mm, verbose=True, new_field=False)  # has little effect and slows down sim

        return propagation_field

    def extract_intensity(self, spatial_window=None, window_sampling=None):
        """
        Extracts intensity in a centered window in the image field.
        Warning - spatial window and window sampling must have same aspect ratio to avoid distortions
        :param spatial_window: height and width of extraction window (tuple) (height,width) (um)
        :param window_sampling: sampling of extraction window (tuple) (y,x)
                                if None returns slice of field without interpolation
        :return extracted_intensity: (nd.array)
        """
        if  (spatial_window is None) and (window_sampling is None):
            intensity = np.abs(self.image_field.u) ** 2

        else:
            if spatial_window is None:
                x_lim = None
                y_lim = None
            else:
                x_lim = (-spatial_window[0],spatial_window[0])
                y_lim = (-spatial_window[1], spatial_window[1])

            extracted_field = self.image_field.cut_resample(x_limits=x_lim,
                                                            y_limits=y_lim,
                                                            num_points=window_sampling,
                                                            new_field=True, interp_kind=(3, 1))
            intensity = np.abs(extracted_field.u) ** 2

        return intensity

    def extract_mask(self):
        # TODO: account for varying sampling size
        x_min = 0
        x_max = 1280
        y_min = (1280 - 1024) // 2
        y_max = 1280 - (1280 - 1024) // 2

        return phase2slm(np.angle(self.mask.u[y_min:y_max, x_min:x_max]))

    def clip_SLM(self, field):
        """
        Clips area outside SLM
        """
        x_min = 0
        x_max = self.sampling_size
        y_min = int(self.sampling_size*0.1)
        y_max = self.sampling_size - int(self.sampling_size*0.1)

        field.u[0:y_min, x_min:x_max] = 0
        field.u[y_max:-1, x_min:x_max] = 0
        field.u[y_min:y_max, 0:x_min] = 0
        field.u[y_min:y_max, x_max:-1] = 0

        return field

    def retrieve_mask(self, target_intensity, num_iterations):
        """
        Finds phase for a given target intensity distribution
        :param target_intensity: (nd.array) 2D array of target intensity with shape matching simulation sampling size
        :param num_iterations: (int) number of iterations for phase retrieval
        :return: phase mask
        """
        print("Starting phase retrieval")

        assert target_intensity.shape == self.mask.u.shape  # Ensure correct target shape
        target_intensity = target_intensity/np.max(target_intensity)  # TODO: Investigate effect of scaling on retrieval

        if self.block_zero_order == True:
            print("Warning: attempting to do phase retrieval while blocking zero order may give poor results")

        SSIM_log = []
        MSE_log = []
        mask_log = []
        intensity_log = []
        for iteration in range(num_iterations):
            self.propagate_forward()  # Forward with input intensity constraint
            image_phase = np.angle(self.image_field.u)

            if iteration % 10 == 0:   # log images
                intensity_log.append(copy.deepcopy(self.image_field))
                mask_log.append(copy.deepcopy(self.mask))

            # metrics and logging
            image_intensity = self.extract_intensity()
            intensity_scaled = image_intensity/np.max(image_intensity)  # Scale for error calc
            SSIM_log.append(calculate_SSIM(intensity_scaled, target_intensity))
            MSE_log.append(calculate_MSE(intensity_scaled, target_intensity))

            self.image_field.u = target_intensity * np.exp(1j * image_phase)  # Target intensity constraint

            object_field = self.propagate_backwards()  # Backward

            clipped_field = self.clip_SLM(object_field)  # clip to SLM

            mask_phase = clipped_field.get_phase(new_field=True)

            if self.supersample_factor > 1: # pixelate mask phase to match SLM pixels if supersampling computation field
                mask_phase.u = phase2complex(pixelate(np.angle(mask_phase.u),self.supersample_factor))

            self.mask = mask_phase  # update SLM with just phase

            print(f"iteration {iteration + 1}: MSE={MSE_log[iteration]}, SSIM={SSIM_log[iteration]}")

        self.propagate_forward()  # propagate through final mask

        print('Phase retrieval done')
        return self.extract_mask(), MSE_log, SSIM_log, mask_log, intensity_log
