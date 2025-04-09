import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
input_folder = "path/to/input_images"  # Replace with your input images folder
output_folder = "path/to/output_images"  # Replace with your output folder

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Define multiple spectrum options with priorities
spectrum_options = [
    {'R': (150, 255), 'G': (200, 255), 'B': (0, 80)},  # Option 1: Most strict
    {'R': (120, 255), 'G': (150, 255), 'B': (0, 100)}, # Option 2: Less strict
    {'R': (100, 255), 'G': (100, 255), 'B': (0, 120)}  # Option 3: Broadest range
]

def apply_dynamic_spectrum(image, spectrum):
    """
    Apply a specific spectrum to the image.
    Args:
    - image: Input image (BGR format).
    - spectrum: Dictionary with 'R', 'G', 'B' thresholds.
    Returns:
    - highlighted_image: Image with highlighted regions based on the spectrum.
    - success: Boolean indicating whether any regions were detected.
    """
    # Split image into channels
    B, G, R = cv2.split(image)
    
    # Create masks based on the spectrum thresholds
    mask_R = (R >= spectrum['R'][0]) & (R <= spectrum['R'][1])
    mask_G = (G >= spectrum['G'][0]) & (G <= spectrum['G'][1])
    mask_B = (B >= spectrum['B'][0]) & (B <= spectrum['B'][1])
    
    combined_mask = mask_R & mask_G & mask_B
    
    # Check if any regions were detected
    success = np.any(combined_mask)
    
    # Highlight the regions with a custom color (default: red)
    highlighted_image = image.copy()
    highlighted_image[combined_mask] = [0, 0, 255]  # Red color in BGR
    
    return highlighted_image, success

def process_images_with_priority(input_folder, output_folder, spectrum_options):
    """
    Process images by applying multiple spectrum options in priority order.
    Args:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to save the processed images.
    - spectrum_options: List of spectrum dictionaries with priorities.
    """
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in tqdm(image_files, desc="Processing Images"):
        # Read the image
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Unable to read {img_file}. Skipping.")
            continue

        # Apply spectrum options in priority order
        for i, spectrum in enumerate(spectrum_options):
            highlighted_image, success = apply_dynamic_spectrum(image, spectrum)
            if success:
                print(f"{img_file}: Detected regions using Spectrum Option {i+1}")
                break  # Stop if a spectrum successfully detects regions
        else:
            print(f"{img_file}: No regions detected with any spectrum options.")
            highlighted_image = image  # Save the original image if no detection
        
        # Save the processed image
        output_path = os.path.join(output_folder, f"processed_{img_file}")
        cv2.imwrite(output_path, highlighted_image)

    print(f"Processing complete. Highlighted images saved to {output_folder}.")

# Run the script
if __name__ == "__main__":
    process_images_with_priority(input_folder, output_folder, spectrum_options)
