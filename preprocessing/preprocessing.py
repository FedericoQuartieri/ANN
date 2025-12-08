"""
Preprocessing Script for Training and Test Datasets
This script applies various preprocessing steps to the training and test data.
"""

import os
import shutil
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION - Toggle preprocessing steps here
# ============================================================================
REMOVE_SHREK = True   # Remove Shrek-contaminated images
FIX_STAINED = True    # Fix green-stained images using background color
SPLIT_DOUBLES = True  # Split double images into two separate images
REMOVE_BLACK_RECT = True  # Remove black rectangles from images
CROP_TO_MASK = False  # Crop images to mask bounding box (removes empty background), can be a problem with padding
RESIZE_AND_NORMALIZE = True  # Resize images and apply normalization/contrast enhancement
DATA_AUGMENTATION = True  # Apply data augmentation (rotation, flipping) - TRAINING SET ONLY

# Crop to mask settings
CROP_PADDING = 10  # Padding around mask bounding box in pixels

# Resize and normalization settings
TARGET_SIZE = 384  # Target size for resizing (will be TARGET_SIZE x TARGET_SIZE)
APPLY_CLAHE = True  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 3.0  # CLAHE clip limit (increased from 2.0 for better contrast)
CLAHE_TILE_GRID_SIZE = (8, 8)  # CLAHE tile grid size

# Data augmentation settings (applied to training set only)
AUGMENT_ROTATIONS = [90, 180, 270]  # Rotation angles in degrees
AUGMENT_FLIPS = []  # Flip types to apply ['horizontal', 'vertical']

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"

# Training data
TRAIN_DATA_DIR = DATA_DIR / "train_data"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"

# Test data
TEST_DATA_DIR = DATA_DIR / "test_data"

# Output directories
PP_DATA_DIR = DATA_DIR / "pp_train_data"
PP_LABELS_PATH = DATA_DIR / "pp_train_labels.csv"
PP_TEST_DATA_DIR = DATA_DIR / "pp_test_data"

# Contamination lists
SHREK_CONTAMINED_PATH = BASE_DIR / "preprocessing" / "shrek_contamined.txt"
STAINED_CONTAMINED_PATH = BASE_DIR / "preprocessing" / "stained_contamined.txt"
DOUBLE_IMAGES_PATH = BASE_DIR / "preprocessing" / "double_images.txt"
BLACK_ADDITION_PATH = BASE_DIR / "preprocessing" / "black_addition.txt"
DOUBLE_IMAGES_TEST_PATH = BASE_DIR / "preprocessing" / "double_images_test.txt"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def load_contamination_list(filepath):
    """Load a list of contaminated image names from a text file (UTF-8 clean format)"""
    if not filepath.exists():
        print(f"Warning: {filepath} not found!")
        return set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            images = set(line.strip() for line in f if line.strip())
        return images
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return set()

def get_mask_filename(img_filename):
    """Convert image filename to corresponding mask filename"""
    return img_filename.replace('img_', 'mask_')

def get_background_color(image):
    """
    Get the most common background color in the image
    
    Args:
        image: BGR image (numpy array)
    
    Returns:
        Background color as BGR tuple
    """
    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Use histogram to find most common color
    # We'll use a simplified approach: get median color
    # (background typically fills most of the image)
    background_color = np.median(pixels, axis=0).astype(np.uint8)
    
    return tuple(background_color)

def detect_tissue_regions(image, white_threshold=240):
    """
    Detect tissue regions in image (non-white areas)
    
    Args:
        image: BGR image (numpy array)
        white_threshold: Threshold to consider pixel as white background
    
    Returns:
        Binary mask where white pixels indicate tissue
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask for non-white pixels (tissue)
    tissue_mask = gray < white_threshold
    
    return tissue_mask.astype(np.uint8) * 255

def find_rectangular_regions(image, min_area_ratio=0.01):
    """
    Find rectangular tissue regions in image (adaptive threshold until separation is clear)
    Handles both white and black backgrounds
    
    Args:
        image: BGR image (numpy array)
        min_area_ratio: Minimum area as ratio of total image area
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...] sorted by area
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    min_area = image.shape[0] * image.shape[1] * min_area_ratio
    
    # Detect background type by checking corner pixels
    corner_values = [
        gray[0, 0], gray[0, -1], 
        gray[-1, 0], gray[-1, -1]
    ]
    avg_corner = np.mean(corner_values)
    is_white_background = avg_corner > 127
    
    # Try different thresholds based on background type
    if is_white_background:
        # White background: tissue is darker
        thresholds = range(255, 200, -5)  # Try from 255 down to 200
        def create_mask(gray, threshold):
            return (gray < threshold).astype(np.uint8) * 255
        def check_tissue(region, threshold):
            return np.sum(region < threshold)
    else:
        # Black background: tissue is brighter
        thresholds = range(0, 55, 5)  # Try from 0 up to 55
        def create_mask(gray, threshold):
            return (gray > threshold).astype(np.uint8) * 255
        def check_tissue(region, threshold):
            return np.sum(region > threshold)
    
    for threshold in thresholds:
        # Create binary mask: non-background = tissue
        tissue_mask = create_mask(gray, threshold)
        
        # Apply minimal morphological operations to clean noise but keep regions separate
        kernel_small = np.ones((3, 3), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Find all contours
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Get significant rectangles
        rectangles = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by minimum area
            if area < min_area:
                continue
            
            # Check if this region contains significant tissue (not mostly background)
            region = gray[y:y+h, x:x+w]
            non_bg_pixels = check_tissue(region, threshold)
            non_bg_ratio = non_bg_pixels / (w * h)
            
            # Only keep regions that have at least 5% non-background pixels
            if non_bg_ratio > 0.05:
                rectangles.append((x, y, w, h, area))
        
        # If we found exactly 2 significant regions, we're done!
        if len(rectangles) == 2:
            rectangles.sort(key=lambda r: r[4], reverse=True)
            return [(x, y, w, h) for x, y, w, h, _ in rectangles]
        
        # If we found more than 2, keep trying different thresholds
        # (might be noise creating extra regions)
        if len(rectangles) > 2:
            continue
    
    # If we exit the loop, we didn't find exactly 2 regions
    # Return empty list to indicate failure
    return []

def split_double_image(image_path, mask_path):
    """
    Split a double image into two separate images by detecting rectangular regions
    Applies aggressive cropping to ensure no background is included
    
    Args:
        image_path: Path to the double image file
        mask_path: Path to the corresponding mask file
    
    Returns:
        List of (image, mask) tuples, or None if splitting fails
    """
    # Read image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        return None
    
    # Find rectangular tissue regions
    rectangles = find_rectangular_regions(image)
    
    # We expect exactly 2 rectangular regions for a double image
    if len(rectangles) < 2:
        return None
    
    # Take the two largest regions
    regions = []
    for i in range(min(2, len(rectangles))):
        x, y, w, h = rectangles[i]
        
        # Apply aggressive inward cropping to ensure no background is caught
        # Crop more from the edges to be safe
        crop_margin = 15  # Crop 15 pixels from each edge
        x_new = x + crop_margin
        y_new = y + crop_margin
        w_new = w - 2 * crop_margin
        h_new = h - 2 * crop_margin
        
        # Ensure we don't crop too much (minimum size check)
        if w_new < 50 or h_new < 50:
            # If cropping would make it too small, use less aggressive cropping
            crop_margin = 5
            x_new = x + crop_margin
            y_new = y + crop_margin
            w_new = w - 2 * crop_margin
            h_new = h - 2 * crop_margin
        
        # Ensure bounds are valid
        x_new = max(0, x_new)
        y_new = max(0, y_new)
        x2 = min(image.shape[1], x_new + w_new)
        y2 = min(image.shape[0], y_new + h_new)
        
        # Extract region from both image and mask
        img_region = image[y_new:y2, x_new:x2].copy()
        mask_region = mask[y_new:y2, x_new:x2].copy()
        
        # Check if mask has any non-zero pixels
        has_content = np.any(mask_region > 0)
        
        regions.append((img_region, mask_region, has_content))
    
    # Filter to keep only regions with non-black masks
    valid_regions = [(img, mask) for img, mask, has_content in regions if has_content]
    
    # Return single valid region if exactly one exists, otherwise None
    if len(valid_regions) == 1:
        return valid_regions
    elif len(valid_regions) == 2:
        # If both have content, return both (shouldn't happen normally)
        return valid_regions
    else:
        # No valid regions or couldn't determine
        return None

def remove_black_rectangle(image):
    """
    Remove black/dark rectangle from image by cropping to tissue region only
    
    Args:
        image: BGR image (numpy array)
    
    Returns:
        Cropped image with black rectangle removed, or original if detection fails
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try progressively higher thresholds to find the tissue (bright) region
    # Dark regions should be < 50, tissue regions should be > 50
    for dark_threshold in range(50, 150, 10):
        # Create binary mask: bright regions = tissue
        tissue_mask = (gray > dark_threshold).astype(np.uint8) * 255
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Get the largest contour (should be the tissue region)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check if this region is significant (at least 20% of image area)
        area_ratio = (w * h) / (image.shape[0] * image.shape[1])
        if area_ratio > 0.2:
            # Crop to this region with small padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            return image[y1:y2, x1:x2]
    
    # If no good region found, return original
    return image

def crop_to_mask_bounding_box(image, mask, padding=10):
    """
    Crop image and mask to the bounding box of the mask's non-zero region
    
    Args:
        image: BGR image (numpy array)
        mask: Grayscale mask (numpy array)
        padding: Additional padding around the bounding box (default: 10 pixels)
    
    Returns:
        Tuple of (cropped_image, cropped_mask) or None if mask has no non-zero pixels
    """
    # Check if mask has any non-zero pixels
    if not np.any(mask > 0):
        return None
    
    # Find all non-zero points in the mask
    coords = cv2.findNonZero(mask)
    
    if coords is None or len(coords) == 0:
        return None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    # Ensure we have valid dimensions
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop both image and mask
    cropped_image = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2].copy()
    
    # Final validation
    if cropped_image.size == 0 or cropped_mask.size == 0:
        return None
    
    return cropped_image, cropped_mask

def resize_and_normalize_image(image, mask, target_size=1024, apply_clahe=True, 
                                clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Resize image and mask to target size while maintaining aspect ratio,
    and apply contrast enhancement via CLAHE
    
    Args:
        image: BGR image (numpy array)
        mask: Grayscale mask (numpy array)
        target_size: Target size for output (will be target_size x target_size)
        apply_clahe: Whether to apply CLAHE for contrast enhancement
        clip_limit: CLAHE clip limit parameter
        tile_grid_size: CLAHE tile grid size parameter
    
    Returns:
        Tuple of (resized_image, resized_mask)
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within target size while maintaining aspect ratio
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image and mask
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Smart background color detection - find the most common light color
    # Sample from corners and edges, but filter out dark colors and extreme colors
    border_thickness = 30
    corner_samples = []
    
    # Sample from all four corners (more representative of true background)
    corner_size = min(border_thickness, min(new_h, new_w) // 4)
    corner_samples.append(resized_image[:corner_size, :corner_size].reshape(-1, 3))  # Top-left
    corner_samples.append(resized_image[:corner_size, -corner_size:].reshape(-1, 3))  # Top-right
    corner_samples.append(resized_image[-corner_size:, :corner_size].reshape(-1, 3))  # Bottom-left
    corner_samples.append(resized_image[-corner_size:, -corner_size:].reshape(-1, 3))  # Bottom-right
    
    all_corner_pixels = np.vstack(corner_samples)
    
    # Convert to grayscale to check brightness
    gray_corners = cv2.cvtColor(all_corner_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
    
    # Filter: keep only pixels that are relatively bright (likely background, not tissue/artifacts)
    # Histopathology backgrounds are typically light (>150 in grayscale)
    bright_mask = gray_corners > 150
    
    if np.sum(bright_mask) > 100:  # If we have enough bright pixels
        # Use only bright pixels for background color
        bright_pixels = all_corner_pixels[bright_mask]
        bg_color = np.round(np.median(bright_pixels, axis=0)).astype(np.uint8)
    else:
        # Fallback: use the brightest pixels available
        brightness_threshold = np.percentile(gray_corners, 75)  # Top 25% brightest
        bright_mask = gray_corners >= brightness_threshold
        bright_pixels = all_corner_pixels[bright_mask]
        bg_color = np.round(np.median(bright_pixels, axis=0)).astype(np.uint8)
    
    # Additional safety: ensure background is not too dark or too saturated
    # If detected color is too dark, default to light gray
    if np.mean(bg_color) < 100:  # Too dark
        bg_color = np.array([240, 240, 240], dtype=np.uint8)  # Light gray default
    
    # Create canvas with uniform background color BEFORE applying CLAHE
    canvas_image = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Calculate padding to center the image
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    
    # Place resized image and mask on canvas BEFORE CLAHE
    canvas_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image
    canvas_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_mask
    
    # Apply CLAHE only to the tissue region (not the padding) to avoid tile artifacts
    if apply_clahe:
        # Create a mask for the tissue region (non-background area)
        tissue_region_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        tissue_region_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = 255
        
        # Convert to LAB color space
        lab = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        
        # Only apply CLAHE to tissue region, keep background unchanged
        l_final = l.copy()
        l_final[tissue_region_mask > 0] = l_clahe[tissue_region_mask > 0]
        
        # Merge channels and convert back to BGR
        lab_final = cv2.merge([l_final, a, b])
        canvas_image = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        # Ensure background remains perfectly uniform by re-filling it
        # This eliminates any artifacts from CLAHE bleeding into padding areas
        background_mask = tissue_region_mask == 0
        canvas_image[background_mask] = bg_color
    
    return canvas_image, canvas_mask

def augment_image(image, mask, rotation=None, flip=None):
    """
    Apply augmentation to image and mask (rotation and/or flipping)
    
    Args:
        image: BGR image (numpy array)
        mask: Grayscale mask (numpy array)
        rotation: Rotation angle in degrees (90, 180, 270) or None
        flip: Flip type ('horizontal', 'vertical') or None
    
    Returns:
        Tuple of (augmented_image, augmented_mask)
    """
    aug_image = image.copy()
    aug_mask = mask.copy()
    
    # Apply rotation
    if rotation == 90:
        aug_image = cv2.rotate(aug_image, cv2.ROTATE_90_CLOCKWISE)
        aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        aug_image = cv2.rotate(aug_image, cv2.ROTATE_180)
        aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_180)
    elif rotation == 270:
        aug_image = cv2.rotate(aug_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Apply flip
    if flip == 'horizontal':
        aug_image = cv2.flip(aug_image, 1)  # 1 = horizontal flip
        aug_mask = cv2.flip(aug_mask, 1)
    elif flip == 'vertical':
        aug_image = cv2.flip(aug_image, 0)  # 0 = vertical flip
        aug_mask = cv2.flip(aug_mask, 0)
    
    return aug_image, aug_mask

def detect_green_stain(image):
    """
    Detect green stain in image using HSV color thresholding
    
    Args:
        image: BGR image (numpy array)
    
    Returns:
        Binary mask where white pixels indicate stain location
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color (adjust these if needed)
    # HSV: Hue [35-85], Saturation [15-255], Value [20-255]
    lower_green = np.array([35, 15, 20])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Optional: morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

def fix_stained_image(image_path, mask_path=None):
    """
    Fix a stained image by replacing green stains with background color
    Also fixes the corresponding mask by setting stained areas to black (0)
    
    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file (optional)
    
    Returns:
        Tuple of (fixed_image, fixed_mask, stain_pixels) or (fixed_image, None, stain_pixels) if no mask
        Returns None if reading fails
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Read mask if provided
    mask = None
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Detect green stain
    stain_mask = detect_green_stain(image)
    
    # Check if any stain was detected
    stain_pixels = np.sum(stain_mask > 0)
    
    if stain_pixels > 0:
        # Get background color
        bg_color = get_background_color(image)
        
        # Create a copy of the image
        fixed_image = image.copy()
        
        # Replace stained pixels with background color in image
        fixed_image[stain_mask > 0] = bg_color
        
        # Fix mask if provided: set stained areas to black (0)
        fixed_mask = None
        if mask is not None:
            fixed_mask = mask.copy()
            fixed_mask[stain_mask > 0] = 0  # Set stained areas to black
        
        return fixed_image, fixed_mask, stain_pixels
    else:
        # No stain detected, return originals
        return image, mask, 0

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def build_exclusion_set(images_to_exclude_sets):
    """
    Build a set of all images to exclude from preprocessing
    
    Args:
        images_to_exclude_sets: List of sets containing image filenames to exclude
    
    Returns:
        Combined set of all images to exclude
    """
    all_excluded = set()
    for img_set in images_to_exclude_sets:
        all_excluded.update(img_set)
    return all_excluded

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_dataset():
    """Main preprocessing pipeline"""
    
    print_section("Dataset Preprocessing Started")
    print(f"Source: {TRAIN_DATA_DIR}")
    print(f"Output: {PP_DATA_DIR}")
    
    # Load training labels
    print("\nLoading training labels...")
    df_original = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded {len(df_original)} samples")
    
    # ========================================================================
    # STEP 1: Build exclusion list
    # ========================================================================
    print_section("Building Exclusion List")
    
    exclusion_sets = []
    total_excluded = set()
    
    # 1a. Shrek contamination
    if REMOVE_SHREK:
        shrek_images = load_contamination_list(SHREK_CONTAMINED_PATH)
        if shrek_images:
            print(f"✓ Shrek contaminated images: {len(shrek_images)}")
            exclusion_sets.append(('Shrek', shrek_images))
            total_excluded.update(shrek_images)
        else:
            print("✗ No Shrek contamination list found")
    else:
        print("⊗ Shrek removal: DISABLED")
    
    # 1b. Stained images (for fixing, not excluding)
    stained_images = set()
    if FIX_STAINED:
        stained_images = load_contamination_list(STAINED_CONTAMINED_PATH)
        if stained_images:
            print(f"✓ Stained images to fix: {len(stained_images)}")
        else:
            print("✗ No stained contamination list found")
    else:
        print("⊗ Stain fixing: DISABLED")
    
    # 1c. Double images (for splitting, not excluding but also not copying directly)
    double_images = set()
    if SPLIT_DOUBLES:
        double_images = load_contamination_list(DOUBLE_IMAGES_PATH)
        if double_images:
            print(f"✓ Double images to split: {len(double_images)}")
        else:
            print("✗ No double images list found")
    else:
        print("⊗ Double image splitting: DISABLED")
    
    # 1d. Black addition images (for cropping black rectangles)
    black_addition_images = set()
    if REMOVE_BLACK_RECT:
        black_addition_images = load_contamination_list(BLACK_ADDITION_PATH)
        if black_addition_images:
            print(f"✓ Black addition images to crop: {len(black_addition_images)}")
        else:
            print("✗ No black addition list found")
    else:
        print("⊗ Black rectangle removal: DISABLED")
    
    print(f"\n>>> TOTAL IMAGES TO EXCLUDE: {len(total_excluded)}")
    
    # ========================================================================
    # STEP 2: Filter labels DataFrame
    # ========================================================================
    print_section("Filtering Dataset")
    
    # Show breakdown by contamination type
    for contamination_type, contaminated_images in exclusion_sets:
        matching = df_original[df_original['sample_index'].isin(contaminated_images)]
        print(f"{contamination_type}: {len(matching)} images will be removed")
    
    # Apply filter - KEEP only images NOT in exclusion list
    df_filtered = df_original[~df_original['sample_index'].isin(total_excluded)].copy()
    
    removed_count = len(df_original) - len(df_filtered)
    print(f"\nOriginal dataset: {len(df_original)} images")
    print(f"Filtered dataset: {len(df_filtered)} images")
    print(f"Removed: {removed_count} images ({removed_count/len(df_original)*100:.2f}%)")
    
    if removed_count == 0:
        print("\n⚠ WARNING: No images were removed! Check your exclusion lists.")
    
    # ========================================================================
    # STEP 3: Copy valid files
    # ========================================================================
    print_section("Copying Preprocessed Files")
    
    # Always clear and recreate output directory to ensure clean state
    if PP_DATA_DIR.exists():
        print(f"✓ Clearing existing directory: {PP_DATA_DIR}")
        shutil.rmtree(PP_DATA_DIR)
    
    PP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created clean directory: {PP_DATA_DIR}")
    
    # Copy/process valid images and masks (those in filtered DataFrame)
    print(f"\nProcessing {len(df_filtered)} valid image-mask pairs...")
    valid_images = set(df_filtered['sample_index'])
    copied_pairs = 0
    fixed_stains = 0
    split_doubles = 0
    cropped_blacks = 0
    missing_files = []
    total_stain_pixels = 0
    
    for idx, img_name in enumerate(valid_images, 1):
        if idx % 200 == 0:
            print(f"  Progress: {idx}/{len(valid_images)}")
        
        src_img = TRAIN_DATA_DIR / img_name
        dst_img = PP_DATA_DIR / img_name
        
        mask_name = get_mask_filename(img_name)
        src_mask = TRAIN_DATA_DIR / mask_name
        dst_mask = PP_DATA_DIR / mask_name
        
        if src_img.exists() and src_mask.exists():
            # Check if this is a double image that needs splitting
            if SPLIT_DOUBLES and img_name in double_images:
                regions = split_double_image(src_img, src_mask)
                if regions is not None and len(regions) >= 1:
                    # Keep only the valid region(s) with non-black masks
                    # Save with original name (no _part suffix since we keep only one)
                    img_region, mask_region = regions[0]
                    cv2.imwrite(str(dst_img), img_region)
                    cv2.imwrite(str(dst_mask), mask_region)
                    
                    split_doubles += 1
                    copied_pairs += 1
                else:
                    # Failed to split, copy original
                    print(f"  ⚠ Could not split {img_name}, copying original")
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_mask, dst_mask)
                    copied_pairs += 1
            # Check if this image needs stain fixing
            elif FIX_STAINED and img_name in stained_images:
                result = fix_stained_image(src_img, src_mask)
                if result is not None:
                    fixed_image, fixed_mask, stain_pixels = result
                    # Save the fixed image
                    cv2.imwrite(str(dst_img), fixed_image)
                    # Save the fixed mask (or copy original if None)
                    if fixed_mask is not None:
                        cv2.imwrite(str(dst_mask), fixed_mask)
                    else:
                        shutil.copy2(src_mask, dst_mask)
                    
                    if stain_pixels > 0:
                        fixed_stains += 1
                        total_stain_pixels += stain_pixels
                else:
                    # Failed to fix, copy originals
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_mask, dst_mask)
                
                copied_pairs += 1
            # Check if this image has black rectangle to remove
            elif REMOVE_BLACK_RECT and img_name in black_addition_images:
                # Read image and mask
                image = cv2.imread(str(src_img))
                mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
                
                if image is not None and mask is not None:
                    # Remove black rectangle from both image and mask
                    cropped_image = remove_black_rectangle(image)
                    cropped_mask = remove_black_rectangle(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                    
                    # Convert mask back to grayscale
                    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2GRAY)
                    
                    # Save cropped versions
                    cv2.imwrite(str(dst_img), cropped_image)
                    cv2.imwrite(str(dst_mask), cropped_mask)
                    cropped_blacks += 1
                else:
                    # Failed to read, copy original
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_mask, dst_mask)
                
                copied_pairs += 1
            else:
                # Copy image and mask normally
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_mask, dst_mask)
                copied_pairs += 1
        else:
            missing_files.append(img_name)
            if not src_img.exists():
                print(f"  ⚠ Missing image: {img_name}")
            if not src_mask.exists():
                print(f"  ⚠ Missing mask: {mask_name}")
    
    print(f"\n✓ Successfully processed {copied_pairs} image-mask pairs")
    if FIX_STAINED and fixed_stains > 0:
        avg_stain_pixels = total_stain_pixels / fixed_stains
        print(f"✓ Fixed stains in {fixed_stains} images (avg {avg_stain_pixels:.0f} pixels/image)")
    if SPLIT_DOUBLES and split_doubles > 0:
        print(f"✓ Split {split_doubles} double images into {split_doubles * 2} separate images")
    if REMOVE_BLACK_RECT and cropped_blacks > 0:
        print(f"✓ Removed black rectangles from {cropped_blacks} images")
    
    if missing_files:
        print(f"✗ Could not copy {len(missing_files)} pairs (missing files)")
    
    # ========================================================================
    # STEP 3a: Crop images to mask bounding box
    # ========================================================================
    if CROP_TO_MASK:
        print_section("Cropping Images to Mask Bounding Box")
        print(f"Padding: {CROP_PADDING} pixels")
        
        # Get all image files in the preprocessed directory
        all_images = sorted([f for f in PP_DATA_DIR.iterdir() if f.name.startswith('img_')])
        
        print(f"\nProcessing {len(all_images)} images for cropping...")
        cropped_count = 0
        discarded_count = 0
        discarded_files = []
        
        for idx, img_path in enumerate(all_images, 1):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(all_images)}")
            
            img_name = img_path.name
            mask_name = get_mask_filename(img_name)
            mask_path = PP_DATA_DIR / mask_name
            
            if img_path.exists() and mask_path.exists():
                # Read image and mask
                image = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if image is not None and mask is not None:
                    # Crop to mask bounding box
                    crop_result = crop_to_mask_bounding_box(image, mask, padding=CROP_PADDING)
                    
                    if crop_result is None:
                        # Mask has no non-zero pixels, discard this image
                        discarded_files.append(img_name)
                        img_path.unlink()  # Delete image
                        mask_path.unlink()  # Delete mask
                        discarded_count += 1
                        continue
                    
                    cropped_image, cropped_mask = crop_result
                    
                    # Validate cropped images are not empty
                    if cropped_image.size == 0 or cropped_mask.size == 0:
                        discarded_files.append(img_name)
                        img_path.unlink()  # Delete image
                        mask_path.unlink()  # Delete mask
                        discarded_count += 1
                        continue
                    
                    # Overwrite with cropped versions
                    cv2.imwrite(str(img_path), cropped_image)
                    cv2.imwrite(str(mask_path), cropped_mask)
                    cropped_count += 1
        
        print(f"\n✓ Cropped {cropped_count} images to mask bounding boxes")
        if discarded_count > 0:
            print(f"✓ Discarded {discarded_count} images with empty masks")
        
        # Remove discarded images from DataFrame
        if discarded_files:
            df_filtered = df_filtered[~df_filtered['sample_index'].isin(discarded_files)].copy()
            print(f"✓ Removed {len(discarded_files)} images with empty masks from labels")
    
    # ========================================================================
    # STEP 3b: Resize and normalize all images
    # ========================================================================
    if RESIZE_AND_NORMALIZE:
        print_section("Resizing and Normalizing Images")
        print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
        print(f"CLAHE contrast enhancement: {'Enabled' if APPLY_CLAHE else 'Disabled'}")
        
        # Get all image files in the preprocessed directory
        all_images = sorted([f for f in PP_DATA_DIR.iterdir() if f.name.startswith('img_')])
        
        print(f"\nProcessing {len(all_images)} images for resizing...")
        resized_count = 0
        
        for idx, img_path in enumerate(all_images, 1):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(all_images)}")
            
            img_name = img_path.name
            mask_name = get_mask_filename(img_name)
            mask_path = PP_DATA_DIR / mask_name
            
            if img_path.exists() and mask_path.exists():
                # Read image and mask
                image = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if image is not None and mask is not None:
                    # Resize and normalize
                    resized_image, resized_mask = resize_and_normalize_image(
                        image, mask, 
                        target_size=TARGET_SIZE,
                        apply_clahe=APPLY_CLAHE,
                        clip_limit=CLAHE_CLIP_LIMIT,
                        tile_grid_size=CLAHE_TILE_GRID_SIZE
                    )
                    
                    # Overwrite with resized versions
                    cv2.imwrite(str(img_path), resized_image)
                    cv2.imwrite(str(mask_path), resized_mask)
                    resized_count += 1
        
        print(f"\n✓ Resized and normalized {resized_count} image-mask pairs")
        if APPLY_CLAHE:
            print(f"✓ Applied CLAHE contrast enhancement (clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID_SIZE})")
    
    # ========================================================================
    # STEP 4: Data Augmentation (Training Set Only)
    # ========================================================================
    if DATA_AUGMENTATION:
        print_section("Data Augmentation")
        print(f"Rotations: {AUGMENT_ROTATIONS}°")
        print(f"Flips: {AUGMENT_FLIPS}")
        
        # Get all current image files
        original_images = sorted([f for f in PP_DATA_DIR.iterdir() if f.name.startswith('img_')])
        original_count = len(original_images)
        
        print(f"\nOriginal dataset size: {original_count} images")
        print(f"Generating augmented images...")
        
        augmented_rows = []  # Store new rows for DataFrame
        augmented_count = 0
        
        for idx, img_path in enumerate(original_images, 1):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{original_count}")
            
            img_name = img_path.name
            mask_name = get_mask_filename(img_name)
            mask_path = PP_DATA_DIR / mask_name
            
            # Get original sample info from DataFrame
            sample_row = df_filtered[df_filtered['sample_index'] == img_name]
            if len(sample_row) == 0:
                continue
            
            original_label = sample_row.iloc[0]['label']
            
            # Read original image and mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                continue
            
            # Generate augmentations
            # Rotations
            for rotation in AUGMENT_ROTATIONS:
                aug_image, aug_mask = augment_image(image, mask, rotation=rotation, flip=None)
                
                # Save augmented image and mask
                base_name = img_name.replace('img_', '').replace('.png', '')
                aug_img_name = f"img_{base_name}_rot{rotation}.png"
                aug_mask_name = f"mask_{base_name}_rot{rotation}.png"
                
                cv2.imwrite(str(PP_DATA_DIR / aug_img_name), aug_image)
                cv2.imwrite(str(PP_DATA_DIR / aug_mask_name), aug_mask)
                
                # Add to DataFrame
                augmented_rows.append({
                    'sample_index': aug_img_name,
                    'label': original_label
                })
                augmented_count += 1
            
            # Flips
            for flip in AUGMENT_FLIPS:
                aug_image, aug_mask = augment_image(image, mask, rotation=None, flip=flip)
                
                # Save augmented image and mask
                base_name = img_name.replace('img_', '').replace('.png', '')
                aug_img_name = f"img_{base_name}_{flip}.png"
                aug_mask_name = f"mask_{base_name}_{flip}.png"
                
                cv2.imwrite(str(PP_DATA_DIR / aug_img_name), aug_image)
                cv2.imwrite(str(PP_DATA_DIR / aug_mask_name), aug_mask)
                
                # Add to DataFrame
                augmented_rows.append({
                    'sample_index': aug_img_name,
                    'label': original_label
                })
                augmented_count += 1
        
        # Add augmented samples to DataFrame
        if augmented_rows:
            df_augmented = pd.DataFrame(augmented_rows)
            df_filtered = pd.concat([df_filtered, df_augmented], ignore_index=True)
        
        final_count = original_count + augmented_count
        if original_count > 0:
            augmentation_factor = final_count / original_count
        else:
            augmentation_factor = 0
        
        print(f"\n✓ Generated {augmented_count} augmented images")
        print(f"✓ Total dataset size: {final_count} images ({augmentation_factor:.1f}x augmentation)")
    
    # ========================================================================
    # STEP 5: Verification
    # ========================================================================
    print_section("Verification")
    
    # Check that excluded images are NOT in output
    verification_failed = False
    for contamination_type, contaminated_images in exclusion_sets:
        sample_to_check = list(contaminated_images)[:5]  # Check first 5
        print(f"\nVerifying {contamination_type} exclusion (checking {len(sample_to_check)} samples):")
        for img_name in sample_to_check:
            exists_in_output = (PP_DATA_DIR / img_name).exists()
            status = "❌ FOUND (ERROR!)" if exists_in_output else "✓ Not found (correct)"
            print(f"  {img_name}: {status}")
            if exists_in_output:
                verification_failed = True
    
    if verification_failed:
        print("\n⚠⚠⚠ VERIFICATION FAILED! Excluded images were found in output!")
    else:
        print("\n✓ Verification passed - excluded images not in output")
    
    # ========================================================================
    # STEP 6: Save preprocessed labels
    # ========================================================================
    print_section("Saving Preprocessed Labels")
    
    # Always overwrite to ensure clean state
    df_filtered.to_csv(PP_LABELS_PATH, index=False)
    print(f"✓ Saved to: {PP_LABELS_PATH}")
    print(f"✓ Total samples: {len(df_filtered)}")
    
    # Verify the file was written correctly
    verify_df = pd.read_csv(PP_LABELS_PATH)
    if len(verify_df) == len(df_filtered):
        print(f"✓ Verification: CSV file is correct")
    else:
        print(f"⚠ Warning: CSV verification failed!")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("Preprocessing Summary")
    print(f"Original dataset: {len(df_original)} samples")
    print(f"Preprocessed dataset: {len(df_filtered)} samples")
    print(f"Removed: {len(df_original) - len(df_filtered)} samples")
    
    print("\nPreprocessing steps applied:")
    for contamination_type, contaminated_images in exclusion_sets:
        print(f"  - {contamination_type}: {len(contaminated_images)} images removed")
    
    if FIX_STAINED and fixed_stains > 0:
        print(f"  - Stained: {fixed_stains} images fixed (not removed)")
    if SPLIT_DOUBLES and split_doubles > 0:
        print(f"  - Doubles: {split_doubles} images cropped to valid region (kept non-black masks)")
    if REMOVE_BLACK_RECT and cropped_blacks > 0:
        print(f"  - Black rectangles: {cropped_blacks} images cropped")
    if CROP_TO_MASK:
        print(f"  - Cropped: Images cropped to mask bounding boxes (padding={CROP_PADDING}px)")
    if RESIZE_AND_NORMALIZE:
        print(f"  - Resized: All images normalized to {TARGET_SIZE}x{TARGET_SIZE}")
        if APPLY_CLAHE:
            print(f"  - CLAHE: Contrast enhancement applied (clip={CLAHE_CLIP_LIMIT})")
    if DATA_AUGMENTATION:
        num_rotations = len(AUGMENT_ROTATIONS)
        num_flips = len(AUGMENT_FLIPS)
        total_aug = num_rotations + num_flips
        if total_aug > 0:
            print(f"  - Augmentation: {total_aug} augmentations per image ({num_rotations} rotations + {num_flips} flips)")
    
    print(f"\nPreprocessed data saved to:")
    print(f"  - Images: {PP_DATA_DIR}")
    print(f"  - Labels: {PP_LABELS_PATH}")
    
    # Label distribution
    print("\nLabel distribution in preprocessed dataset:")
    label_counts = df_filtered['label'].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(df_filtered) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    print_section("Preprocessing Complete")
    
    if verification_failed:
        print("⚠ NOTE: Verification found issues - please review the output above")
        return False
    return True

def preprocess_test_dataset():
    """
    Preprocess the test dataset (no labels, no stain/shrek removal)
    """
    print_section("Test Dataset Preprocessing")
    print(f"Source: {TEST_DATA_DIR}")
    print(f"Output: {PP_TEST_DATA_DIR}")
    
    # ========================================================================
    # STEP 1: Load double images list
    # ========================================================================
    print_section("Loading Configuration")
    
    double_images = set()
    if SPLIT_DOUBLES:
        double_images = load_contamination_list(DOUBLE_IMAGES_TEST_PATH)
        if double_images:
            print(f"✓ Double images to split: {len(double_images)}")
        else:
            print("✗ No double images list found")
    else:
        print("⊗ Double image splitting: DISABLED")
    
    # ========================================================================
    # STEP 2: Process test files
    # ========================================================================
    print_section("Processing Test Files")
    
    # Always clear and recreate output directory to ensure clean state
    if PP_TEST_DATA_DIR.exists():
        print(f"✓ Clearing existing directory: {PP_TEST_DATA_DIR}")
        shutil.rmtree(PP_TEST_DATA_DIR)
    
    PP_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created clean directory: {PP_TEST_DATA_DIR}")
    
    # Get all image files in test directory
    all_images = sorted([f for f in TEST_DATA_DIR.iterdir() if f.name.startswith('img_')])
    
    print(f"\nProcessing {len(all_images)} test images...")
    copied_pairs = 0
    split_doubles = 0
    missing_files = []
    
    for idx, src_img in enumerate(all_images, 1):
        if idx % 200 == 0:
            print(f"  Progress: {idx}/{len(all_images)}")
        
        img_name = src_img.name
        dst_img = PP_TEST_DATA_DIR / img_name
        
        mask_name = get_mask_filename(img_name)
        src_mask = TEST_DATA_DIR / mask_name
        dst_mask = PP_TEST_DATA_DIR / mask_name
        
        if src_img.exists() and src_mask.exists():
            # Check if this is a double image that needs splitting
            if SPLIT_DOUBLES and img_name in double_images:
                regions = split_double_image(src_img, src_mask)
                if regions is not None and len(regions) >= 1:
                    # Keep only the valid region with non-black mask
                    img_region, mask_region = regions[0]
                    cv2.imwrite(str(dst_img), img_region)
                    cv2.imwrite(str(dst_mask), mask_region)
                    
                    split_doubles += 1
                    copied_pairs += 1
                else:
                    # Failed to split, copy original
                    print(f"  ⚠ Could not split {img_name}, copying original")
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_mask, dst_mask)
                    copied_pairs += 1
            else:
                # Copy image and mask normally
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_mask, dst_mask)
                copied_pairs += 1
        else:
            missing_files.append(img_name)
            if not src_img.exists():
                print(f"  ⚠ Missing image: {img_name}")
            if not src_mask.exists():
                print(f"  ⚠ Missing mask: {mask_name}")
    
    print(f"\n✓ Successfully processed {copied_pairs} image-mask pairs")
    if SPLIT_DOUBLES and split_doubles > 0:
        print(f"✓ Split {split_doubles} double images (kept non-black mask regions)")
    if missing_files:
        print(f"✗ Could not copy {len(missing_files)} pairs (missing files)")
    
    # ========================================================================
    # STEP 2a: Crop images to mask bounding box
    # ========================================================================
    if CROP_TO_MASK:
        print_section("Cropping Images to Mask Bounding Box")
        print(f"Padding: {CROP_PADDING} pixels")
        
        # Get all image files in the preprocessed directory
        all_images = sorted([f for f in PP_TEST_DATA_DIR.iterdir() if f.name.startswith('img_')])
        
        print(f"\nProcessing {len(all_images)} images for cropping...")
        cropped_count = 0
        discarded_count = 0
        
        for idx, img_path in enumerate(all_images, 1):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(all_images)}")
            
            img_name = img_path.name
            mask_name = get_mask_filename(img_name)
            mask_path = PP_TEST_DATA_DIR / mask_name
            
            if img_path.exists() and mask_path.exists():
                # Read image and mask
                image = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if image is not None and mask is not None:
                    # Crop to mask bounding box
                    crop_result = crop_to_mask_bounding_box(image, mask, padding=CROP_PADDING)
                    
                    if crop_result is None:
                        # Mask has no non-zero pixels, discard this image
                        img_path.unlink()  # Delete image
                        mask_path.unlink()  # Delete mask
                        discarded_count += 1
                        continue
                    
                    cropped_image, cropped_mask = crop_result
                    
                    # Validate cropped images are not empty
                    if cropped_image.size == 0 or cropped_mask.size == 0:
                        img_path.unlink()  # Delete image
                        mask_path.unlink()  # Delete mask
                        discarded_count += 1
                        continue
                    
                    # Overwrite with cropped versions
                    cv2.imwrite(str(img_path), cropped_image)
                    cv2.imwrite(str(mask_path), cropped_mask)
                    cropped_count += 1
        
        print(f"\n✓ Cropped {cropped_count} images to mask bounding boxes")
        if discarded_count > 0:
            print(f"✓ Discarded {discarded_count} test images with empty masks")
    
    # ========================================================================
    # STEP 2b: Resize and normalize all images
    # ========================================================================
    if RESIZE_AND_NORMALIZE:
        print_section("Resizing and Normalizing Images")
        print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
        print(f"CLAHE contrast enhancement: {'Enabled' if APPLY_CLAHE else 'Disabled'}")
        
        # Get all image files in the preprocessed directory
        all_images = sorted([f for f in PP_TEST_DATA_DIR.iterdir() if f.name.startswith('img_')])
        
        print(f"\nProcessing {len(all_images)} images for resizing...")
        resized_count = 0
        
        for idx, img_path in enumerate(all_images, 1):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(all_images)}")
            
            img_name = img_path.name
            mask_name = get_mask_filename(img_name)
            mask_path = PP_TEST_DATA_DIR / mask_name
            
            if img_path.exists() and mask_path.exists():
                # Read image and mask
                image = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if image is not None and mask is not None:
                    # Resize and normalize
                    resized_image, resized_mask = resize_and_normalize_image(
                        image, mask, 
                        target_size=TARGET_SIZE,
                        apply_clahe=APPLY_CLAHE,
                        clip_limit=CLAHE_CLIP_LIMIT,
                        tile_grid_size=CLAHE_TILE_GRID_SIZE
                    )
                    
                    # Overwrite with resized versions
                    cv2.imwrite(str(img_path), resized_image)
                    cv2.imwrite(str(mask_path), resized_mask)
                    resized_count += 1
        
        print(f"\n✓ Resized and normalized {resized_count} image-mask pairs")
        if APPLY_CLAHE:
            print(f"✓ Applied CLAHE contrast enhancement (clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID_SIZE})")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("Test Preprocessing Summary")
    print(f"Total test images processed: {copied_pairs}")
    
    print("\nPreprocessing steps applied:")
    if SPLIT_DOUBLES and split_doubles > 0:
        print(f"  - Doubles: {split_doubles} images cropped to valid region (kept non-black masks)")
    if CROP_TO_MASK:
        print(f"  - Cropped: Images cropped to mask bounding boxes (padding={CROP_PADDING}px)")
    if RESIZE_AND_NORMALIZE:
        print(f"  - Resized: All images normalized to {TARGET_SIZE}x{TARGET_SIZE}")
        if APPLY_CLAHE:
            print(f"  - CLAHE: Contrast enhancement applied (clip={CLAHE_CLIP_LIMIT})")
    
    print(f"\nPreprocessed test data saved to: {PP_TEST_DATA_DIR}")
    
    print_section("Test Preprocessing Complete")
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" PREPROCESSING PIPELINE")
    print("="*80)
    print("\nThis script will preprocess both training and test datasets.\n")
    
    # Preprocess training dataset
    print("\n" + "#"*80)
    print("# PART 1: TRAINING DATASET")
    print("#"*80)
    train_success = preprocess_dataset()
    
    # Preprocess test dataset
    print("\n\n" + "#"*80)
    print("# PART 2: TEST DATASET")
    print("#"*80)
    test_success = preprocess_test_dataset()
    
    # Final summary
    print("\n\n" + "="*80)
    print(" PREPROCESSING PIPELINE COMPLETE")
    print("="*80)
    print(f"Training dataset: {'✓ Success' if train_success else '✗ Failed'}")
    print(f"Test dataset: {'✓ Success' if test_success else '✗ Failed'}")
    print("="*80)
