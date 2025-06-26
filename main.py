from PIL import Image
import numpy as np
from numpy.fft import fft2, fftshift

def measure_high_frequency_content(image_path):
    """
    Measures the high-frequency content of an image using the Fourier Transform.

    Args:
        image_path (str): The path to the JPEG image file.

    Returns:
        float: A measure of high-frequency content (e.g., sum of magnitudes
               in the high-frequency regions of the Fourier spectrum).
    """
    try:
        # Open the image and convert to grayscale for simpler processing
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

        # Perform 2D Fast Fourier Transform (FFT)
        f_transform = fft2(img_array)
        # Shift the zero-frequency component to the center for visualization
        f_transform_shifted = fftshift(f_transform)

        # Calculate the magnitude spectrum
        magnitude_spectrum = np.abs(f_transform_shifted)

        # Define a region for high frequencies (e.g., outer regions of the spectrum)
        rows, cols = img_array.shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates

        # Define a radius for the low-frequency cutoff
        cutoff_radius = min(crow, ccol) // 4  # Adjust as needed

        # Create a mask for high-frequency regions
        mask = np.zeros_like(magnitude_spectrum, dtype=bool)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if distance > cutoff_radius:
                    mask[i, j] = True

        # Sum the magnitudes in the high-frequency regions
        high_frequency_sum = np.sum(magnitude_spectrum[mask])

        return high_frequency_sum

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
# Assuming 'your_image.jpg' is in the same directory
# high_freq_measure = measure_high_frequency_content('your_image.jpg')
# if high_freq_measure is not None:
#     print(f"High-frequency content measure: {high_freq_measure}")
