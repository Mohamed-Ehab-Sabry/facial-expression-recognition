import os
import cv2
import numpy as np


def validate_grayscale(image):
    """Ensure the image is 2D (grayscale). If RGB/BGR, convert to grayscale."""
    if image is None:
        return None
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_clahe_opencv(image, clip_limit=1.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_unsharp_masking(image, kernel_size=(5, 5), k=0.3):
    """Apply unsharp masking with the selected best settings."""
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    mask = image.astype(np.float32) - blurred.astype(np.float32)
    sharpened = image.astype(np.float32) + k * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_image_array(image):
    """Pipeline: validate grayscale -> CLAHE -> unsharp masking (5x5, k=0.3)."""
    image = validate_grayscale(image)
    if image is None:
        return None

    image = apply_clahe_opencv(image)
    image = apply_unsharp_masking(image, kernel_size=(5, 5), k=0.3)
    return image


def preprocess_image_file(input_path, output_path=None):
    """Preprocess a single image file and optionally save it."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    processed = preprocess_image_array(image)

    if processed is None:
        raise ValueError(f"Could not read image: {input_path}")

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, processed)

    return processed


def preprocess_dataset(source_root, target_root):
    """Preprocess all images in a folder tree and preserve subfolder structure."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for root, _, files in os.walk(source_root):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_ext:
                continue

            input_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(input_path, source_root)
            output_path = os.path.join(target_root, rel_path)

            processed = preprocess_image_file(input_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, processed)

    print("Preprocessing complete.")


def visualize_before_after(sample_image_paths, source_root, target_root, save_path="preprocessing_comparison.png"):
    """Display and save 5 before/after comparisons using already processed outputs."""
    try:
        import importlib
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc

    plt.figure(figsize=(10, 15))

    for i, original_path in enumerate(sample_image_paths[:5]):
        rel_path = os.path.relpath(original_path, source_root)
        processed_path = os.path.join(target_root, rel_path)

        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

        if original is None or processed is None:
            continue

        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(original, cmap="gray")
        plt.title(f"Original {i + 1}")
        plt.axis("off")

        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(processed, cmap="gray")
        plt.title(f"Processed {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # preprocess_dataset("fer2013/versions/1/train", "fer2013/versions/1/train_preprocessed")
    # preprocess_dataset("fer2013/versions/1/test", "fer2013/versions/1/test_preprocessed")
    pass
