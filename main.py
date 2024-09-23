import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology
from skimage.io import imread_collection

class ImageProcessor:
    def __init__(self, image_dir='processed_images/*.png'):
        """
        Initialize the ImageProcessor with a directory containing images.

        Parameters:
        - image_dir (str): Directory path with images (supports glob patterns).
        """
        # Check if the provided directory exists
        if not os.path.exists(os.path.dirname(image_dir)):
            # Try an alternative path if the initial one doesn't exist
            image_dir = '/app/projects_mkosno/euc_gumed_densities/processed_images/*.png'
            if not os.path.exists(os.path.dirname(image_dir)):
                raise FileNotFoundError("Image directory not found.")
        self.image_dir = image_dir
        self.images = None
        self.current_image = None
        self.gray_image = None
        self.rescaled_image = None
        self.bright_mask = None
        self.morph_mask = None
        self.final_mask = None
        self.cropped_image_rgba = None
        self.ratio = None

    def load_images(self):
        """
        Load images from the specified directory.
        """
        self.images = imread_collection(self.image_dir)
        print(f"Loaded {len(self.images)} images from {self.image_dir}")

    def select_image(self, index=0):
        """
        Select an image from the loaded images by index.

        Parameters:
        - index (int): Index of the image to select.
        """
        if self.images is None:
            raise ValueError("Images not loaded. Call load_images() first.")
        if index < 0 or index >= len(self.images):
            raise IndexError("Index out of range.")
        self.current_image = self.images[index]
        print(f"Selected image at index {index}")

    def preprocess_image(self):
        """
        Convert the image to grayscale and rescale pixel values.
        """
        if self.current_image is None:
            raise ValueError("No image selected. Call select_image() first.")

        # Convert image to RGB if it has an alpha channel
        if self.current_image.shape[2] == 4:
            rgb_image = color.rgba2rgb(self.current_image)
        else:
            rgb_image = self.current_image

        # Convert to grayscale
        self.gray_image = color.rgb2gray(rgb_image)

        # Rescale pixel values to 0-255
        min_val = np.min(self.gray_image)
        max_val = np.max(self.gray_image)
        self.rescaled_image = ((self.gray_image - min_val) / (max_val - min_val)) * 255
        self.rescaled_image = self.rescaled_image.astype(np.uint8)
        print("Image preprocessed: pixel values rescaled to 0-255.")

    def create_bright_mask(self, threshold=0.5):
        """
        Create a binary mask of bright areas in the image based on a threshold.

        Parameters:
        - threshold (float): Threshold value between 0 and 1.
        """
        if self.gray_image is None:
            raise ValueError("Gray image not available. Call preprocess_image() first.")
        self.bright_mask = self.gray_image > threshold
        print(f"Bright mask created with threshold {threshold}.")

    def apply_morphology(self, disk_size=30):
        """
        Apply morphological operations to the bright mask.

        Parameters:
        - disk_size (int): Size of the structuring element.
        """
        if self.bright_mask is None:
            raise ValueError("Bright mask not available. Call create_bright_mask() first.")
        selem = morphology.disk(disk_size)
        # Apply erosion followed by closing
        eroded_mask = morphology.erosion(self.bright_mask, selem)
        self.morph_mask = morphology.closing(eroded_mask, selem)
        print(f"Morphological operations applied with disk size {disk_size}.")

    def remove_contours(self):
        """
        Remove contours from the morphological mask to refine the mask.
        """
        if self.morph_mask is None:
            raise ValueError("Morphological mask not available. Call apply_morphology() first.")

        # Convert mask to uint8
        mask_uint8 = self.morph_mask.astype(np.uint8) * 255

        # Threshold the mask
        _, thresh_white = cv2.threshold(mask_uint8, 240, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(255 - mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill contours
        for contour in contours:
            cv2.drawContours(thresh_white, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

        self.final_mask = thresh_white == 0
        print("Contours removed from the mask.")

    def create_cropped_image(self):
        """
        Create an RGBA image using the rescaled image and the final mask.
        """
        if self.final_mask is None:
            raise ValueError("Final mask not available. Call remove_contours() first.")
        if self.rescaled_image is None:
            raise ValueError("Rescaled image not available. Call preprocess_image() first.")

        # Ensure rescaled image is 3-channel
        if len(self.rescaled_image.shape) == 2:
            rescaled_rgb = np.stack([self.rescaled_image]*3, axis=-1)
        else:
            rescaled_rgb = self.rescaled_image

        # Create alpha channel
        alpha_channel = self.final_mask.astype(np.uint8) * 255

        # Combine to create RGBA image
        self.cropped_image_rgba = np.dstack((rescaled_rgb, alpha_channel))
        print("Cropped RGBA image created.")

    def calculate_pixel_ratio(self, white_threshold=100, black_threshold=0):
        """
        Calculate the ratio of white to black pixels in the cropped image.

        Parameters:
        - white_threshold (int): Threshold to consider a pixel as white.
        - black_threshold (int): Threshold to consider a pixel as black.
        """
        if self.cropped_image_rgba is None:
            raise ValueError("Cropped image not available. Call create_cropped_image() first.")

        # Flatten the image for easier computation
        flat_image = self.cropped_image_rgba.reshape(-1, self.cropped_image_rgba.shape[-1])

        # Count white pixels (pixels where all RGB channels are above the threshold)
        num_white_pixels = np.sum(np.all(flat_image[:, :3] >= white_threshold, axis=1))

        # Count black pixels (pixels where all RGB channels are below the threshold)
        num_black_pixels = np.sum(np.all(flat_image[:, :3] <= black_threshold, axis=1))

        # Calculate ratio
        self.ratio = num_white_pixels / num_black_pixels if num_black_pixels > 0 else float('inf')
        print(f"White pixels: {num_white_pixels}, Black pixels: {num_black_pixels}, Ratio: {self.ratio}")

    def display_results(self):
        """
        Display intermediate and final images using matplotlib.
        """
        if self.rescaled_image is None or self.bright_mask is None:
            raise ValueError("Required images not available. Ensure previous steps have been completed.")

        # Display rescaled image and bright mask
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.rescaled_image, cmap='gray')
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(self.bright_mask, cmap='gray')
        axs[1].set_title('Bright Areas Detected by Threshold')
        axs[1].axis('off')
        plt.show()

        # Display morphological masks
        if self.morph_mask is not None and self.final_mask is not None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(self.morph_mask, cmap='gray')
            axs[0].set_title('Morphological Mask')
            axs[0].axis('off')

            axs[1].imshow(self.final_mask, cmap='gray')
            axs[1].set_title('Final Mask after Contour Removal')
            axs[1].axis('off')
            plt.show()

        # Display final cropped image
        if self.cropped_image_rgba is not None:
            plt.figure(figsize=(6, 6))
            plt.imshow(self.cropped_image_rgba)
            plt.title('Cropped RGBA Image')
            plt.axis('off')
            plt.show()

    def save_result_image(self, file_path):
        """
        Save the final cropped RGBA image to the specified file path.

        Parameters:
        - file_path (str): The path where the image will be saved.
        """
        if self.cropped_image_rgba is None:
            raise ValueError("Cropped image not available. Call create_cropped_image() first.")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert RGBA image to uint8
        image_to_save = self.cropped_image_rgba.astype(np.uint8)

        # Save the image using matplotlib's imsave
        plt.imsave(file_path, image_to_save)
        print(f"Image saved to {file_path}")

# Example usage:
processor = ImageProcessor()
processor.load_images()
processor.select_image(index=1040)
processor.preprocess_image()
processor.create_bright_mask()
processor.apply_morphology()
processor.remove_contours()
processor.create_cropped_image()
processor.calculate_pixel_ratio()
processor.display_results()
processor.save_result_image('app/documents/test.png')
