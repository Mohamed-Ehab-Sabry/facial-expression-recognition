import os
from modules.preprocessing import preprocess_dataset, visualize_before_after


def get_sample_images(source_root, max_samples=5):
    """Collect up to max_samples image paths from the source dataset."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    sample_paths = []

    for root, _, files in os.walk(source_root):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in valid_ext:
                sample_paths.append(os.path.join(root, file_name))
                if len(sample_paths) == max_samples:
                    return sample_paths

    return sample_paths


def main():
    source_root = "../fer2013/versions/1"
    target_root = "../fer2013/versions/1_preprocessed"

    print("Starting dataset preprocessing...")
    preprocess_dataset(source_root, target_root)

    print("Selecting sample images for before/after visualization...")
    sample_image_paths = get_sample_images(source_root, max_samples=5)

    if len(sample_image_paths) < 5:
        print(f"Warning: Only found {len(sample_image_paths)} images for visualization.")

    if sample_image_paths:
        visualize_before_after(
            sample_image_paths=sample_image_paths,
            source_root=source_root,
            target_root=target_root,
            save_path="preprocessing_comparison.png",
        )
        print("Visualization complete. Saved as preprocessing_comparison.png")
    else:
        print("No images found to visualize.")


if __name__ == "__main__":
    main()
