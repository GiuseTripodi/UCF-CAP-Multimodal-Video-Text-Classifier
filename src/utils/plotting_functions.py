from os.path import join
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def plot_class_distribution(df_train, df_val, df_test, output_dir):
    """
    Plot the class distribution for each split and combined.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Train distribution
    train_counts = df_train['label'].value_counts().sort_index()
    axes[0, 0].bar(range(len(train_counts)), train_counts.values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Train Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Class Index')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Val distribution
    val_counts = df_val['label'].value_counts().sort_index()
    axes[0, 1].bar(range(len(val_counts)), val_counts.values, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Validation Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Class Index')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Test distribution
    test_counts = df_test['label'].value_counts().sort_index()
    axes[1, 0].bar(range(len(test_counts)), test_counts.values, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Class Index')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Combined distribution
    all_labels = pd.concat([df_train['label'], df_val['label'], df_test['label']])
    all_counts = all_labels.value_counts().sort_index()
    axes[1, 1].bar(range(len(all_counts)), all_counts.values, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_title('Combined - Class Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Class Index')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: class_distribution.png")
    plt.close()


def plot_samples_per_class(df_train, df_val, df_test, output_dir, top_n=20):
    """
    Plot the top N classes by number of samples.
    """
    all_labels = pd.concat([df_train['label'], df_val['label'], df_test['label']])
    class_counts = all_labels.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(class_counts)), class_counts.values, color='teal', edgecolor='black')
    ax.set_yticks(range(len(class_counts)))
    ax.set_yticklabels(class_counts.index)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Class Label', fontsize=12)
    ax.set_title(f'Top {top_n} Classes by Sample Count', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f' {int(width)}',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(join(output_dir, 'top_classes_by_samples.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: top_classes_by_samples.png")
    plt.close()


def plot_split_distribution(df_train, df_val, df_test, output_dir):
    """
    Plot the distribution of samples across splits.
    """
    split_sizes = [len(df_train), len(df_val), len(df_test)]
    split_names = ['Train', 'Validation', 'Test']
    colors = ['#FF9999', '#66B2FF', '#99FF99']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(split_sizes, labels=split_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    bars = ax2.bar(split_names, split_sizes, color=colors, edgecolor='black')
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Samples per Split', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(join(output_dir, 'split_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: split_distribution.png")
    plt.close()


def plot_caption_length_distribution(df_train, df_val, df_test, output_dir):
    """
    Plot caption length distributions across splits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Train
    axes[0, 0].hist(df_train['caption_length'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Train - Caption Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Caption Length (words)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df_train['caption_length'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_train['caption_length'].mean():.1f}")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Val
    axes[0, 1].hist(df_val['caption_length'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Validation - Caption Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Caption Length (words)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df_val['caption_length'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_val['caption_length'].mean():.1f}")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Test
    axes[1, 0].hist(df_test['caption_length'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Test - Caption Length Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Caption Length (words)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(df_test['caption_length'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_test['caption_length'].mean():.1f}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Combined boxplot
    data_to_plot = [df_train['caption_length'], df_val['caption_length'], df_test['caption_length']]
    bp = axes[1, 1].boxplot(data_to_plot, labels=['Train', 'Val', 'Test'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['skyblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    axes[1, 1].set_title('Caption Length Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Caption Length (words)')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(output_dir, 'caption_length_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: caption_length_distribution.png")
    plt.close()


def plot_class_distribution_per_split(df_train, df_val, df_test, output_dir):
    """
    Compare class distribution across all three splits side by side.
    """
    # Get all unique classes
    all_classes = sorted(set(df_train['label'].unique()) |
                         set(df_val['label'].unique()) |
                         set(df_test['label'].unique()))

    train_counts = df_train['label'].value_counts()
    val_counts = df_val['label'].value_counts()
    test_counts = df_test['label'].value_counts()

    # Create aligned data
    train_data = [train_counts.get(cls, 0) for cls in all_classes]
    val_data = [val_counts.get(cls, 0) for cls in all_classes]
    test_data = [test_counts.get(cls, 0) for cls in all_classes]

    x = range(len(all_classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar([i - width for i in x], train_data, width, label='Train', color='skyblue', edgecolor='black')
    ax.bar(x, val_data, width, label='Validation', color='lightgreen', edgecolor='black')
    ax.bar([i + width for i in x], test_data, width, label='Test', color='lightcoral', edgecolor='black')

    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=90, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(output_dir, 'class_distribution_per_split.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: class_distribution_per_split.png")
    plt.close()

def analyze_dataset_statistics(df_train, df_val, df_test, output_dir):
    """
    Generate and save comprehensive statistics about the dataset splits.
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Basic statistics
    print(f"\nTotal samples:")
    print(f"  Train: {len(df_train)}")
    print(f"  Val:   {len(df_val)}")
    print(f"  Test:  {len(df_test)}")
    print(f"  Total: {len(df_train) + len(df_val) + len(df_test)}")

    # Class statistics
    print(f"\nNumber of unique classes:")
    print(f"  Train: {df_train['label'].nunique()}")
    print(f"  Val:   {df_val['label'].nunique()}")
    print(f"  Test:  {df_test['label'].nunique()}")

    # Caption statistics
    df_train['caption_length'] = df_train['caption'].str.split().str.len()
    df_val['caption_length'] = df_val['caption'].str.split().str.len()
    df_test['caption_length'] = df_test['caption'].str.split().str.len()

    print(f"\nCaption length statistics (words):")
    print(f"  Train - Mean: {df_train['caption_length'].mean():.2f}, Median: {df_train['caption_length'].median():.0f}")
    print(f"  Val   - Mean: {df_val['caption_length'].mean():.2f}, Median: {df_val['caption_length'].median():.0f}")
    print(f"  Test  - Mean: {df_test['caption_length'].mean():.2f}, Median: {df_test['caption_length'].median():.0f}")

    # Save statistics to file
    with open(join(output_dir, 'dataset_statistics.txt'), 'w') as f:
        f.write("DATASET STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}\n")
        f.write(
            f"Unique classes: Train={df_train['label'].nunique()}, Val={df_val['label'].nunique()}, Test={df_test['label'].nunique()}\n")
        f.write(
            f"\nCaption lengths (mean): Train={df_train['caption_length'].mean():.2f}, Val={df_val['caption_length'].mean():.2f}, Test={df_test['caption_length'].mean():.2f}\n")


