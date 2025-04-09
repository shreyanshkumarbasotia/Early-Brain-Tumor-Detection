import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
input_folder = "path/to/input_images"  # Replace with your input images folder
output_folder = "path/to/output_images"  # Replace with your output folder

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

def highlight_bright_areas_with_color(image, highlight_color=(0, 0, 255)):
    """
    Highlight the brightest areas with a custom RGB color on top of the original tumor region.
    Args:
    - image: Input image (BGR format).
    - highlight_color: Tuple for the highlight color (default is red: (0, 0, 255)).
    """
    # Convert to grayscale for intensity calculation
    intensity = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]

    # Create a mask for bright areas
    bright_mask = (intensity > 200) & (image[:, :, 2] > 200) & (image[:, :, 1] > 200) & (image[:, :, 0] > 200)

    # Overlay the bright areas with the specified color
    highlighted_image = image.copy()
    highlighted_image[bright_mask] = highlight_color  # Use the custom color

    return highlighted_image

def process_images(input_folder, output_folder, highlight_color=(0, 0, 255)):
    """
    Process all images in the input folder by highlighting the brightest areas with a custom color.
    Save the outputs to the output folder.
    Args:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to save the processed images.
    - highlight_color: Tuple for the highlight color (default is red: (0, 0, 255)).
    """
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in tqdm(image_files, desc="Processing Images"):
        # Read image
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Unable to read {img_file}. Skipping.")
            continue

        # Highlight the brightest areas with the specified color
        highlighted_image = highlight_bright_areas_with_color(image, highlight_color)

        # Save the processed image
        output_path = os.path.join(output_folder, f"highlighted_{img_file}")
        cv2.imwrite(output_path, highlighted_image)

    print(f"Processing complete. Highlighted images saved to {output_folder}.")

# Run the script
if __name__ == "__main__":
    # Define the highlight color (default is red)
    highlight_color = (0, 0, 255)  # BGR format: (Blue, Green, Red)

    process_images(input_folder, output_folder, highlight_color)
