"""
Data Verification Script
========================

This script verifies the processed CelebA dataset splits and displays summary statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
data_path = r"c:\Users\deepa rajesh\OneDrive\Desktop\faceauth\data"
processed_path = os.path.join(data_path, "processed")

def load_and_verify_splits():
    """Load and verify the dataset splits"""
    print("=" * 50)
    print("CelebA Dataset Verification")
    print("=" * 50)
    
    # Load splits
    train_df = pd.read_csv(os.path.join(processed_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_path, "test.csv"))
    val_df = pd.read_csv(os.path.join(processed_path, "val.csv"))
    metadata_df = pd.read_csv(os.path.join(processed_path, "metadata.csv"))
    stats_df = pd.read_csv(os.path.join(processed_path, "dataset_stats.csv"))
    
    print(f"✓ Train set: {len(train_df):,} samples")
    print(f"✓ Test set: {len(test_df):,} samples")  
    print(f"✓ Validation set: {len(val_df):,} samples")
    print(f"✓ Total: {len(metadata_df):,} samples")
    print()
    
    # Verify split proportions
    total = len(train_df) + len(test_df) + len(val_df)
    train_pct = len(train_df) / total * 100
    test_pct = len(test_df) / total * 100
    val_pct = len(val_df) / total * 100
    
    print("Split Proportions:")
    print(f"  Train: {train_pct:.1f}% (target: 70%)")
    print(f"  Test: {test_pct:.1f}% (target: 20%)")
    print(f"  Validation: {val_pct:.1f}% (target: 10%)")
    print()
    
    # Get attribute columns
    attr_columns = [col for col in train_df.columns 
                   if col not in ['image_id', 'partition', 'split', 'image_path']]
    
    print(f"Number of attributes: {len(attr_columns)}")
    print()
    
    # Check attribute distribution for key attributes
    key_attrs = ['Male', 'Young', 'Smiling', 'Eyeglasses', 'Heavy_Makeup']
    
    print("Key Attribute Distributions:")
    print("-" * 40)
    for attr in key_attrs:
        if attr in attr_columns:
            train_pos = (train_df[attr] == 1).sum() / len(train_df) * 100
            test_pos = (test_df[attr] == 1).sum() / len(test_df) * 100
            val_pos = (val_df[attr] == 1).sum() / len(val_df) * 100
            
            print(f"{attr}:")
            print(f"  Train: {train_pos:.1f}%")
            print(f"  Test: {test_pos:.1f}%")
            print(f"  Val: {val_pos:.1f}%")
            print()
    
    # Verify image paths exist (sample check)
    print("Image Path Verification (sample check):")
    print("-" * 40)
    sample_imgs = train_df['image_path'].head(5)
    
    for img_path in sample_imgs:
        # Convert relative path to absolute
        full_path = os.path.join(data_path, "..", img_path.replace("data\\", "").replace("data/", ""))
        full_path = os.path.normpath(full_path)
        
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"{status} {os.path.basename(img_path)}")
    
    print("\n" + "=" * 50)
    print("Dataset verification complete!")
    print("=" * 50)
    
    return {
        'train_df': train_df,
        'test_df': test_df, 
        'val_df': val_df,
        'metadata_df': metadata_df,
        'stats_df': stats_df,
        'attr_columns': attr_columns
    }

def plot_attribute_distributions(data_dict):
    """Plot attribute distributions across splits"""
    train_df = data_dict['train_df']
    test_df = data_dict['test_df']
    val_df = data_dict['val_df']
    attr_columns = data_dict['attr_columns']
    
    # Key attributes to visualize
    key_attrs = ['Male', 'Young', 'Smiling', 'Eyeglasses', 'Heavy_Makeup', 
                'Black_Hair', 'Blond_Hair', 'Attractive', 'Wearing_Lipstick']
    
    # Calculate positive ratios for each split
    splits_data = []
    for split_name, df in [('Train', train_df), ('Test', test_df), ('Val', val_df)]:
        for attr in key_attrs:
            if attr in attr_columns:
                pos_ratio = (df[attr] == 1).sum() / len(df)
                splits_data.append({
                    'Split': split_name,
                    'Attribute': attr,
                    'Positive_Ratio': pos_ratio
                })
    
    splits_df = pd.DataFrame(splits_data)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=splits_df, x='Attribute', y='Positive_Ratio', hue='Split')
    plt.title('Attribute Distribution Across Train/Test/Val Splits')
    plt.ylabel('Positive Ratio')
    plt.xlabel('Attributes')
    plt.xticks(rotation=45)
    plt.legend(title='Split')
    plt.tight_layout()
    plt.show()
    
    return splits_df

if __name__ == "__main__":
    # Run verification
    data_dict = load_and_verify_splits()
    
    # Plot distributions
    print("\\nGenerating attribute distribution plots...")
    splits_df = plot_attribute_distributions(data_dict)
    
    print("\\nVerification complete! Your CelebA dataset is ready for training.")
    print("\\nNext steps:")
    print("1. Load the processed data using the CSV files")
    print("2. Implement your explainable face recognition model")
    print("3. Use the attribute labels for explainability")