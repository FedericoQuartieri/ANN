import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data/train_data'
LABELS_FILE = 'data/train_labels.csv'
GRAPHS_DIR = 'analysis/graphs'
SAMPLE_SIZE = None  # Set to number for subset, None for all images

# Create output directory
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load labels and image information"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load labels
    df = pd.read_csv(LABELS_FILE)
    print(f"✓ Loaded {len(df)} samples with labels")
    print(f"✓ Classes: {df['label'].unique().tolist()}")
    
    return df


def analyze_class_distribution(df):
    """1. CLASS DISTRIBUTION ANALYSIS"""
    print("\n" + "=" * 80)
    print("1. CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Count per class
    class_counts = df['label'].value_counts()
    class_percentages = df['label'].value_counts(normalize=True) * 100
    
    # Print statistics
    print("\nClass Distribution:")
    for label in class_counts.index:
        count = class_counts[label]
        pct = class_percentages[label]
        print(f"  {label:20s}: {count:4d} samples ({pct:5.2f}%)")
    
    # Calculate imbalance metrics
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Metrics:")
    print(f"  Max class size: {max_count}")
    print(f"  Min class size: {min_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print("  ⚠ WARNING: Significant class imbalance detected! Consider:")
        print("    - Class weights during training")
        print("    - Oversampling minority classes")
        print("    - Data augmentation focused on minority classes")
    else:
        print("  ✓ Classes are relatively balanced")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    ax1 = axes[0]
    colors = sns.color_palette("husl", len(class_counts))
    bars = ax1.bar(range(len(class_counts)), class_counts.values, color=colors)
    ax1.set_xticks(range(len(class_counts)))
    ax1.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({class_percentages.values[i]:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    ax2 = axes[1]
    ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {GRAPHS_DIR}/01_class_distribution.png")


def analyze_image_properties(df):
    """2. IMAGE PROPERTIES AND STATISTICS"""
    print("\n" + "=" * 80)
    print("2. IMAGE PROPERTIES ANALYSIS")
    print("=" * 80)
    
    # Collect image properties
    widths, heights, aspects = [], [], []
    channels_list = []
    file_sizes = []
    
    sample_df = df.sample(n=SAMPLE_SIZE) if SAMPLE_SIZE else df
    
    print(f"\nAnalyzing {len(sample_df)} images...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Loading images"):
        img_path = os.path.join(DATA_DIR, row['sample_index'])
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspects.append(w / h)
            channels_list.append(len(img.getbands()))
            file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
    
    # Statistics
    print(f"\nImage Dimension Statistics:")
    print(f"  Width  - Min: {min(widths):4d}, Max: {max(widths):4d}, Mean: {np.mean(widths):6.1f}, Std: {np.std(widths):6.1f}")
    print(f"  Height - Min: {min(heights):4d}, Max: {max(heights):4d}, Mean: {np.mean(heights):6.1f}, Std: {np.std(heights):6.1f}")
    print(f"  Aspect - Min: {min(aspects):5.2f}, Max: {max(aspects):5.2f}, Mean: {np.mean(aspects):5.2f}")
    
    unique_dims = set(zip(widths, heights))
    print(f"\nUnique dimensions: {len(unique_dims)}")
    
    if len(unique_dims) == 1:
        print(f"  ✓ All images have same dimensions: {list(unique_dims)[0]}")
    else:
        print(f"  ⚠ Images have varying dimensions - resizing will be required")
        print(f"  Most common dimensions:")
        dim_counts = pd.Series([f"{w}x{h}" for w, h in zip(widths, heights)]).value_counts()
        for dim, count in dim_counts.head(5).items():
            print(f"    {dim}: {count} images")
    
    print(f"\nColor Channels:")
    channel_counts = pd.Series(channels_list).value_counts()
    for ch, count in channel_counts.items():
        print(f"  {ch} channels: {count} images")
    
    print(f"\nFile Size Statistics:")
    print(f"  Min: {min(file_sizes):6.1f} KB")
    print(f"  Max: {max(file_sizes):6.1f} KB")
    print(f"  Mean: {np.mean(file_sizes):6.1f} KB")
    print(f"  Total: {sum(file_sizes)/1024:6.1f} MB")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Width distribution
    ax1 = axes[0, 0]
    ax1.hist(widths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(widths):.0f}')
    ax1.set_xlabel('Width (pixels)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Image Width Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Height distribution
    ax2 = axes[0, 1]
    ax2.hist(heights, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(heights):.0f}')
    ax2.set_xlabel('Height (pixels)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Image Height Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Aspect ratio
    ax3 = axes[1, 0]
    ax3.hist(aspects, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(aspects), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aspects):.2f}')
    ax3.set_xlabel('Aspect Ratio (W/H)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Scatter: Width vs Height
    ax4 = axes[1, 1]
    ax4.scatter(widths, heights, alpha=0.5, s=20, color='purple')
    ax4.set_xlabel('Width (pixels)', fontsize=11)
    ax4.set_ylabel('Height (pixels)', fontsize=11)
    ax4.set_title('Width vs Height Scatter', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/02_image_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {GRAPHS_DIR}/02_image_dimensions.png")
    
    return widths, heights, channels_list

def analyze_data_quality(df, sample_size=100):
    """3. DATA QUALITY ANALYSIS - OUTLIERS AND NOISE"""
    print("\n" + "=" * 80)
    print("3. DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    brightness_vals = []
    contrast_vals = []
    sharpness_vals = []
    file_info = []
    
    print(f"\nAnalyzing quality metrics for {len(sample_df)} images...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Quality analysis"):
        img_path = os.path.join(DATA_DIR, row['sample_index'])
        
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            
            if len(img.shape) == 3:
                # Brightness (mean pixel value)
                brightness = img.mean()
                brightness_vals.append(brightness)
                
                # Contrast (std of pixel values)
                contrast = img.std()
                contrast_vals.append(contrast)
                
                # Sharpness approximation (using Laplacian variance)
                gray = img.mean(axis=2) if len(img.shape) == 3 else img
                laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                
                # Simple convolution
                from scipy.ndimage import convolve
                edges = convolve(gray.astype(float), laplacian)
                sharpness = edges.var()
                sharpness_vals.append(sharpness)
                
                file_info.append({
                    'file': row['sample_index'],
                    'brightness': brightness,
                    'contrast': contrast,
                    'sharpness': sharpness,
                    'label': row['label']
                })
    
    # Detect outliers using IQR method
    def detect_outliers_iqr(data, name):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        pct = (len(outliers) / len(data)) * 100
        print(f"  {name:15s}: {len(outliers):3d} outliers ({pct:5.2f}%) - Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        return lower_bound, upper_bound, outliers
    
    print(f"\nOutlier Detection (IQR method):")
    bright_lower, bright_upper, bright_outliers = detect_outliers_iqr(brightness_vals, "Brightness")
    contrast_lower, contrast_upper, contrast_outliers = detect_outliers_iqr(contrast_vals, "Contrast")
    sharp_lower, sharp_upper, sharp_outliers = detect_outliers_iqr(sharpness_vals, "Sharpness")
    
    # Identify potentially problematic images
    quality_df = pd.DataFrame(file_info)
    
    problematic = quality_df[
        (quality_df['brightness'] < bright_lower) | (quality_df['brightness'] > bright_upper) |
        (quality_df['contrast'] < contrast_lower) | (quality_df['contrast'] > contrast_upper) |
        (quality_df['sharpness'] < sharp_lower) | (quality_df['sharpness'] > sharp_upper)
    ]
    
    print(f"\nPotentially Problematic Images: {len(problematic)} ({len(problematic)/len(quality_df)*100:.1f}%)")
    if len(problematic) > 0 and len(problematic) <= 10:
        print("  Files:")
        for _, row in problematic.iterrows():
            print(f"    - {row['file']}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    avg_brightness = np.mean(brightness_vals)
    avg_contrast = np.mean(contrast_vals)
    
    if avg_brightness < 50:
        print("  ⚠ Images are generally dark - consider brightness adjustment")
    elif avg_brightness > 200:
        print("  ⚠ Images are generally bright - may need adjustment")
    else:
        print("  ✓ Brightness levels appear normal")
    
    if avg_contrast < 30:
        print("  ⚠ Low contrast detected - consider histogram equalization")
    else:
        print("  ✓ Contrast levels appear adequate")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Brightness distribution with outliers
    ax1 = axes[0, 0]
    ax1.hist(brightness_vals, bins=30, color='gold', edgecolor='black', alpha=0.7)
    ax1.axvline(bright_lower, color='red', linestyle='--', linewidth=2, label='Lower bound')
    ax1.axvline(bright_upper, color='red', linestyle='--', linewidth=2, label='Upper bound')
    ax1.axvline(np.mean(brightness_vals), color='blue', linestyle='-', linewidth=2, label='Mean')
    ax1.set_xlabel('Mean Brightness', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Brightness Distribution with Outlier Bounds', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Contrast distribution with outliers
    ax2 = axes[0, 1]
    ax2.hist(contrast_vals, bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax2.axvline(contrast_lower, color='red', linestyle='--', linewidth=2, label='Lower bound')
    ax2.axvline(contrast_upper, color='red', linestyle='--', linewidth=2, label='Upper bound')
    ax2.axvline(np.mean(contrast_vals), color='blue', linestyle='-', linewidth=2, label='Mean')
    ax2.set_xlabel('Contrast (Std Dev)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Contrast Distribution with Outlier Bounds', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Sharpness distribution
    ax3 = axes[1, 0]
    ax3.hist(sharpness_vals, bins=30, color='cyan', edgecolor='black', alpha=0.7)
    ax3.axvline(sharp_lower, color='red', linestyle='--', linewidth=2, label='Lower bound')
    ax3.axvline(sharp_upper, color='red', linestyle='--', linewidth=2, label='Upper bound')
    ax3.axvline(np.mean(sharpness_vals), color='blue', linestyle='-', linewidth=2, label='Mean')
    ax3.set_xlabel('Sharpness Score', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Sharpness Distribution with Outlier Bounds', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Brightness vs Contrast scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(brightness_vals, contrast_vals, 
                         c=sharpness_vals, cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Brightness', fontsize=11)
    ax4.set_ylabel('Contrast', fontsize=11)
    ax4.set_title('Brightness vs Contrast (colored by Sharpness)', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Sharpness', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/03_data_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {GRAPHS_DIR}/03_data_quality.png")


def create_sample_visualization(df, n_samples=12):
    """4. SAMPLE IMAGES VISUALIZATION"""
    print("\n" + "=" * 80)
    print("4. CREATING SAMPLE VISUALIZATION")
    print("=" * 80)
    
    # Sample images from each class
    samples_per_class = n_samples // len(df['label'].unique())
    sampled_images = []
    
    for label in df['label'].unique():
        class_samples = df[df['label'] == label].sample(n=min(samples_per_class, len(df[df['label'] == label])))
        sampled_images.append(class_samples)
    
    sampled_df = pd.concat(sampled_images).sample(frac=1).head(n_samples)
    
    # Create grid
    cols = 4
    rows = (len(sampled_df) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes]
    
    print(f"\nCreating visualization with {len(sampled_df)} sample images...")
    for idx, (ax, (_, row)) in enumerate(zip(axes, sampled_df.iterrows())):
        img_path = os.path.join(DATA_DIR, row['sample_index'])
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{row['label']}\n{row['sample_index']}", fontsize=9, fontweight='bold')
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(sampled_df), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/04_sample_images.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {GRAPHS_DIR}/04_sample_images.png")


def main():
    """Main execution"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "MEDICAL IMAGE DATASET - EDA REPORT" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Load data
    df = load_data()
    
    # Run analyses
    analyze_class_distribution(df)
    analyze_image_properties(df)
    analyze_data_quality(df, sample_size=100)
    create_sample_visualization(df, n_samples=12)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n✓ All analyses completed successfully!")
    print(f"✓ Generated 4 visualization files in '{GRAPHS_DIR}/' directory")
    print(f"\nGenerated files:")
    print(f"  1. 01_class_distribution.png - Class imbalance analysis")
    print(f"  2. 02_image_dimensions.png - Image size and aspect ratios")
    print(f"  3. 03_data_quality.png - Quality metrics and outliers")
    print(f"  4. 04_sample_images.png - Visual samples from dataset")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
