# Re-importing necessary libraries and redefining the function since the code execution state was reset

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

def RFPattern(x, y, C, s, r0, A, w, phi):
    # Convert cartesian coordinates to polar coordinates
    theta = np.arctan2(y, x)  # Polar angle
    r = np.sqrt(x**2 + y**2)  # Radius

    # Radius of the deformed pattern in radians
    ro = r0 * (1 + A * np.sin(w * theta + phi))

    # Radial fourth derivative of a Gaussian
    D4 = C * (1 - 4 * ((r - ro) / s)**2 + 4/3 * ((r - ro) / s)**4) * np.exp(-((r - ro) / s)**2)
    
    # Normalize D4 for visualization
    D4_normalized = (D4 - D4.min()) / (D4.max() - D4.min())
    
    # Create alpha channel based on a radius threshold
    alpha = np.zeros_like(r)
    radius_threshold = r0 + 2 * s  # Define the radius threshold beyond which we make it transparent
    alpha[r <= radius_threshold] = 1
    
    # Combine into an RGBA image
    rgba_image = np.zeros((D4.shape[0], D4.shape[1], 4))  # Empty RGBA image
    rgba_image[..., 0] = D4_normalized  # Red channel
    rgba_image[..., 1] = D4_normalized  # Green channel
    rgba_image[..., 2] = D4_normalized  # Blue channel
    rgba_image[..., 3] = alpha  # Alpha channel
    
    return rgba_image

def save_individual_patterns(params, filename_base, x, y):
    for i, (C, s, r0, A, w, phi) in enumerate(params, start=1):
        rgba_image = RFPattern(x, y, C, s, r0, A, w, phi)  # This now expects RGBA image data
        
        # Convert numpy array to image
        img = Image.fromarray((rgba_image * 255).astype(np.uint8), 'RGBA')

        # Save the image directly using PIL, which handles RGBA properly
        filename = f'{filename_base}_{i}.png'
        img.save(filename)

        # Yield the filename for potential downstream use
        yield filename

# Parameters for different shapes
shapes_params = [
    # (C, s, r0, A, w, phi)
    (1, 1, 6, 0.40, 3, 0),
    (1, 1, 6, 0.30, 3, 0), 
    (1, 1, 6, 0.20, 3, 0), 
    (1, 1, 6, 0.10, 3, 0), 
    (1, 1, 6, 0.10, 4, 0),
    (1, 1, 6, 0.10, 5, 0), 
    (1, 1, 6, 0.20, 5, 0), 
    (1, 1, 6, 0.30, 5, 0), 
    (1, 1, 6, 0.40, 5, 0) 
]

# Directory and base filename for the PNG files
output_dir = 'images/'
filename_base = output_dir + 'pattern'

# Generate the mesh grid
x, y = np.meshgrid(np.linspace(-10, 10, 400), np.linspace(-10, 10, 400))
if not os.path.isdir(output_dir ):
    os.makedirs(output_dir )

# Generate and save each pattern
pattern_files = list(save_individual_patterns(shapes_params, filename_base, x, y))
pattern_files