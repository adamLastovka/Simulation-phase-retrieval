import diffractio
from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from matplotlib import cm
import torch

diffractio.config.CONF_DRAWING['color_intensity'] = cm.jet

"""
simulation module converted such that:
1. mask and image_fields are tensors
2. RS propagation uses pytorch functions that support gradient tracking
"""

def phase2complex(mask_phase):
    return torch.exp(1j * mask_phase)


def slm2phase(mask_slm):
    return (mask_slm / 210.0) * (2 * np.pi) - np.pi


def phase2slm(mask_phase):
    return (mask_phase + np.pi) / (2 * np.pi) * 210.0

def complex2intensity(field):
    return torch.abs(field)**2
def kernelRS(X, Y, wavelength, z, n=1):
    """Kernel for RS propagation.

    Parameters:
        X (torch.tensor): positions x
        Y (torch.tensor): positions y
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refraction index of background
        kind (str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex torch.tensor: kernel
    """
    k = 2 * np.pi * n / wavelength
    R = torch.sqrt(X ** 2 + Y ** 2 + z ** 2)

    return 1 / (2 * np.pi) * torch.exp(1.j * k * R) * z / R ** 2 * (1 / R - 1.j * k)

def kernelRSinverse(X, Y, wavelength=0.6328 * um, z=-10 * mm, n=1):
    """Kernel for inverse RS propagation

    Parameters:
        X(numpy.array): positions x
        Y(numpy.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refraction index of background
        kind(str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel
    """
    k = 2 * np.pi * n / wavelength
    R = torch.sqrt(X**2 + Y**2 + z**2)

    return 1 / (2 * np.pi) * torch.exp(-1.j * k * R) * z / R * (1 / R + 1.j * k)


def RS(u, z, n, wavelength, x, y, xout=None, yout=None):
    if xout is None: # Can remove xout and yout
        xout = x[0]
    if yout is None:
        yout = y[0]

    xout = x + xout - x[0]
    yout = y + yout - y[0]

    nx = len(xout)
    ny = len(yout)
    dx = xout[1] - xout[0]
    dy = yout[1] - yout[0]

    precise = 0
    W = 1

    U = torch.zeros((2 * ny - 1, 2 * nx - 1), dtype=torch.complex64) # padding
    U[0:ny, 0:nx] = W * u

    xext = x[0] - xout.flip(0)
    xext = xext[0:-1]
    xext = torch.cat((xext, x - xout[0]))

    yext = y[0] - yout.flip(0)
    yext = yext[0:-1]
    yext = torch.cat((yext, y - yout[0]))

    Xext, Yext = torch.meshgrid(xext, yext)

    if z > 0:
        H = kernelRS(Xext, Yext, wavelength, z, n)
    else:
        H = kernelRSinverse(Xext, Yext, wavelength, z, n)

    S = torch.fft.ifft2(torch.fft.fft2(U) * torch.fft.fft2(H)) * dx * dy
    Usalida = S[ny - 1:, nx - 1:]

    return Usalida


class PropagationSimulation:
    def __init__(self, wavelength, computation_window, sampling_size, mask_read):
        self.wavelength = wavelength

        x_size = computation_window[1]  # Spatial dimensions
        y_size = computation_window[0]

        x_vect = np.linspace(-x_size / 2, x_size / 2, sampling_size)  # Vectors of sample points
        y_vect = np.linspace(-y_size / 2, y_size / 2, sampling_size)

        # Source intensity
        u0 = Scalar_source_XY(x=x_vect, y=y_vect, wavelength=wavelength)
        u0.gauss_beam((0, 0), (6 * mm, 6 * mm), (0, 0))

        # SLM MASK Embedding
        self.mask = torch.empty((sampling_size,sampling_size),dtype=torch.complex64)  # create mask scalar field
        mask_read = slm2phase(torch.from_numpy(mask_read))  # convert SLM to complex mask
        self.embed_mask(mask_read)  # embed mask in center

        # Lens masks
        lens_1 = Scalar_mask_XY(x=x_vect, y=y_vect, wavelength=wavelength)
        lens_1.lens_spherical(r0=(0, 0), focal=700 * mm, radius=30 * mm, refraction_index=1.5)

        lens_2 = Scalar_mask_XY(x=x_vect, y=y_vect, wavelength=wavelength)
        lens_2.lens_spherical(r0=(0, 0), focal=160 * mm, radius=30 * mm, refraction_index=1.5)

        # Backwards lens masks
        lens_1b = lens_1.conjugate(new_field=True)
        lens_2b = lens_2.conjugate(new_field=True)

        self.sampling_size = sampling_size
        self.x_size = x_size
        self.y_size = y_size
        self.x_vect = torch.from_numpy(x_vect)
        self.y_vect = torch.from_numpy(y_vect)
        self.u0 = torch.from_numpy(u0.u)
        self.lens_1 = torch.from_numpy(lens_1.u)
        self.lens_2 = torch.from_numpy(lens_2.u)
        self.lens_1b = torch.from_numpy(lens_1b.u)
        self.lens_2b = torch.from_numpy(lens_2b.u)
        self.image_field = torch.empty((1280, 1024))

    def propagate_forward(self):
        u2 = self.u0 * self.mask  # SLM
        # u1.RS(z=100*mm, verbose=True) # has little effect and slows down sim
        u3 = u2 * self.lens_1
        u4 = RS(u3, 1050 * mm, 1, self.wavelength, self.x_vect, self.y_vect)
        u5 = u4 * self.lens_2  # F-Theta
        self.image_field = RS(u5, 285 * mm, 1, self.wavelength, self.x_vect, self.y_vect)

    def embed_mask(self, mask_phase):
        if torch.max(mask_phase) > np.pi+1e-3 or torch.min(mask_phase) < -np.pi-1e-3:  # TODO: Fix mask phases exceeding maxima due to precision
            print(f"Mask exceeds maxima - max:{torch.max(mask_phase):.2e} min:{torch.min(mask_phase):.2e}")

        # TODO: implement embedding based on spatial dimensions
        x_min = 0
        x_max = 1280
        y_min = (1280 - 1024) // 2
        y_max = 1280 - (1280 - 1024) // 2

        self.mask[y_min:y_max, x_min:x_max] = phase2complex(mask_phase)

    def extract_intensity(self):
        # TODO: to implement extracting based on spatial dimensions
        x_min = 0
        x_max = 1280
        y_min = (1280 - 1024) // 2
        y_max = 1280 - (1280 - 1024) // 2

        return torch.abs(self.image_field[y_min:y_max, x_min:x_max]) ** 2

    def draw_mask(self):
        extents = (self.x_vect[0],self.x_vect[-1],self.y_vect[0],self.y_vect[-1])

        plt.imshow(torch.angle(self.mask),
                   interpolation='nearest',
                   aspect='auto',
                   origin='lower',
                   extent=extents)

        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.title("Phase mask")
        plt.set_cmap('jet')
        plt.axis(extents)
        plt.axis('scaled')
        plt.show()

    def draw_image(self,kind):
        extents = (self.x_vect[0], self.x_vect[-1], self.y_vect[0], self.y_vect[-1])

        if kind == 'intensity':
            plt.imshow(complex2intensity(self.image_field),
                       interpolation='nearest',
                       aspect='auto',
                       origin='lower',
                       extent=extents)
            plt.title("Image field intensity")

        elif kind == 'phase':
            plt.imshow(torch.angle(self.image_field),
                       interpolation='nearest',
                       aspect='auto',
                       origin='lower',
                       extent=extents)
            plt.title("Image field phase")
        else:
            raise Exception("Wrong plot king")

        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.set_cmap('jet')
        plt.axis(extents)
        plt.axis('scaled')
        plt.show()
