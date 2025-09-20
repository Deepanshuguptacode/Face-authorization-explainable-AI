"""
CelebA Dataset Preprocessing Script
==================================

This script preprocesses the CelebA dataset for explainable face recognition:
1. Loads the full CelebA dataset metadata
2. Samples 50% of the dataset while maintaining attribute distribution
3. Splits the sampled data into train (70%), test (20%), and validation (10%) sets
4. Saves processed metadata files for each split

Dataset Info:
- Total images: ~202,599
- Attributes: 40 facial attributes (binary: -1/1)
- Target: Use 50% (~101,300 images)
- Splits: Train (~70,910), Test (~20,260), Val (~10,130)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CelebAPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the CelebA preprocessor
        
        Args:
            data_path (str): Path to the raw CelebA data directory
        """
        self.data_path = data_path
        self.raw_path = os.path.join(data_path, 'raw', 'celebA')
        self.processed_path = os.path.join(data_path, 'processed')
        self.img_path = os.path.join(self.raw_path, 'img_align_celeba', 'img_align_celeba')
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Dataset configuration
        self.sample_ratio = 0.5  # Use 50% of the dataset
        self.train_ratio = 0.7   # 70% for training
        self.test_ratio = 0.2    # 20% for testing  
        self.val_ratio = 0.1     # 10% for validation
        
        # Random seed for reproducibility
        self.random_state = 42
        np.random.seed(self.random_state)
        
    def load_metadata(self):
        """Load CelebA metadata files"""
        logger.info("Loading CelebA metadata files...")
        
        # Load attributes
        attr_file = os.path.join(self.raw_path, 'list_attr_celeba.csv')
        self.attributes_df = pd.read_csv(attr_file)
        logger.info(f"Loaded attributes for {len(self.attributes_df)} images")
        logger.info(f"Found {len(self.attributes_df.columns)-1} attributes")
        
        # Load partition info (original CelebA splits)
        partition_file = os.path.join(self.raw_path, 'list_eval_partition.csv')
        self.partition_df = pd.read_csv(partition_file)
        logger.info(f"Loaded partition info for {len(self.partition_df)} images")
        
        # Merge attributes and partition info
        self.full_df = pd.merge(self.attributes_df, self.partition_df, on='image_id')
        logger.info(f"Merged dataset shape: {self.full_df.shape}")
        
        # Get attribute columns (exclude image_id and partition)
        self.attribute_columns = [col for col in self.full_df.columns 
                                if col not in ['image_id', 'partition']]
        logger.info(f"Attribute columns: {len(self.attribute_columns)}")
        
        return self.full_df
    
    def verify_images_exist(self, df):
        """Verify that image files exist for the given dataframe"""
        logger.info("Verifying image files exist...")
        missing_images = []
        
        for idx, row in df.iterrows():
            img_file = os.path.join(self.img_path, row['image_id'])
            if not os.path.exists(img_file):
                missing_images.append(row['image_id'])
        
        if missing_images:
            logger.warning(f"Found {len(missing_images)} missing images")
            # Remove missing images from dataframe
            df = df[~df['image_id'].isin(missing_images)]
            logger.info(f"Dataset after removing missing images: {len(df)}")
        else:
            logger.info("All image files exist!")
            
        return df
    
    def sample_dataset(self, df):
        """Sample 50% of the dataset while maintaining attribute distribution"""
        logger.info(f"Sampling {self.sample_ratio*100}% of the dataset...")
        
        original_size = len(df)
        target_size = int(original_size * self.sample_ratio)
        
        logger.info(f"Original dataset size: {original_size}")
        logger.info(f"Target sample size: {target_size}")
        
        # Use stratified sampling based on a few key attributes to maintain diversity
        # We'll use gender (Male) and a few other attributes for stratification
        key_attributes = ['Male', 'Young', 'Smiling', 'Eyeglasses', 'Heavy_Makeup']
        
        # Create a stratification key by combining key attributes
        stratify_key = df[key_attributes].apply(lambda x: ''.join(x.astype(str)), axis=1)
        
        # Perform stratified sampling
        try:
            sampled_df, _ = train_test_split(
                df, 
                test_size=1-self.sample_ratio,
                stratify=stratify_key,
                random_state=self.random_state
            )
        except ValueError:
            # If stratified sampling fails due to small groups, use simple random sampling
            logger.warning("Stratified sampling failed, using random sampling")
            sampled_df = df.sample(n=target_size, random_state=self.random_state)
        
        logger.info(f"Sampled dataset size: {len(sampled_df)}")
        
        # Log attribute distribution comparison
        self._log_attribute_distribution(df, sampled_df, "Original vs Sampled")
        
        return sampled_df
    
    def create_splits(self, df):
        """Create train/test/validation splits"""
        logger.info("Creating train/test/validation splits...")
        
        # First split: separate train from test+val
        train_size = self.train_ratio
        temp_size = self.test_ratio + self.val_ratio
        
        # Use stratified sampling based on key attributes
        key_attributes = ['Male', 'Young', 'Smiling', 'Eyeglasses']
        stratify_key = df[key_attributes].apply(lambda x: ''.join(x.astype(str)), axis=1)
        
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_size,
                stratify=stratify_key,
                random_state=self.random_state
            )
        except ValueError:
            logger.warning("Stratified sampling failed for train split, using random sampling")
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_size,
                random_state=self.random_state
            )
        
        # Second split: separate test from validation
        test_relative_size = self.test_ratio / temp_size
        
        try:
            temp_stratify_key = temp_df[key_attributes].apply(lambda x: ''.join(x.astype(str)), axis=1)
            test_df, val_df = train_test_split(
                temp_df,
                test_size=1-test_relative_size,
                stratify=temp_stratify_key,
                random_state=self.random_state
            )
        except ValueError:
            logger.warning("Stratified sampling failed for test/val split, using random sampling")
            test_df, val_df = train_test_split(
                temp_df,
                test_size=1-test_relative_size,
                random_state=self.random_state
            )
        
        logger.info(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        
        return train_df, test_df, val_df
    
    def save_splits(self, train_df, test_df, val_df, full_sampled_df):
        """Save the splits to CSV files"""
        logger.info("Saving splits to CSV files...")
        
        # Add split information
        train_df = train_df.copy()
        test_df = test_df.copy()
        val_df = val_df.copy()
        
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        val_df['split'] = 'val'
        
        # Add full image paths
        for df in [train_df, test_df, val_df]:
            df['image_path'] = df['image_id'].apply(
                lambda x: os.path.join('data', 'raw', 'celebA', 'img_align_celeba', 'img_align_celeba', x)
            )
        
        # Save individual split files
        train_df.to_csv(os.path.join(self.processed_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_path, 'test.csv'), index=False)
        val_df.to_csv(os.path.join(self.processed_path, 'val.csv'), index=False)
        
        # Save combined metadata
        full_sampled_df_with_split = pd.concat([train_df, test_df, val_df], ignore_index=True)
        full_sampled_df_with_split.to_csv(os.path.join(self.processed_path, 'metadata.csv'), index=False)
        
        logger.info(f"Saved train.csv: {len(train_df)} samples")
        logger.info(f"Saved test.csv: {len(test_df)} samples")
        logger.info(f"Saved val.csv: {len(val_df)} samples")
        logger.info(f"Saved metadata.csv: {len(full_sampled_df_with_split)} samples")
        
        return full_sampled_df_with_split
    
    def _log_attribute_distribution(self, original_df, sampled_df, comparison_name):
        """Log attribute distribution comparison"""
        logger.info(f"\n=== {comparison_name} Attribute Distribution ===")
        
        # Compare distributions for key attributes
        key_attrs = ['Male', 'Young', 'Smiling', 'Eyeglasses', 'Heavy_Makeup', 'Black_Hair', 'Blond_Hair']
        
        for attr in key_attrs:
            if attr in original_df.columns:
                orig_pos = (original_df[attr] == 1).sum()
                orig_neg = (original_df[attr] == -1).sum()
                samp_pos = (sampled_df[attr] == 1).sum()
                samp_neg = (sampled_df[attr] == -1).sum()
                
                orig_ratio = orig_pos / (orig_pos + orig_neg) * 100
                samp_ratio = samp_pos / (samp_pos + samp_neg) * 100
                
                logger.info(f"{attr}: Original {orig_ratio:.1f}% -> Sampled {samp_ratio:.1f}%")
    
    def generate_summary_stats(self, train_df, test_df, val_df):
        """Generate and save summary statistics"""
        logger.info("Generating summary statistics...")
        
        stats = {
            'total_samples': len(train_df) + len(test_df) + len(val_df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'val_samples': len(val_df),
            'train_ratio': len(train_df) / (len(train_df) + len(test_df) + len(val_df)),
            'test_ratio': len(test_df) / (len(train_df) + len(test_df) + len(val_df)),
            'val_ratio': len(val_df) / (len(train_df) + len(test_df) + len(val_df)),
            'num_attributes': len(self.attribute_columns)
        }
        
        # Attribute statistics for each split
        for split_name, split_df in [('train', train_df), ('test', test_df), ('val', val_df)]:
            for attr in self.attribute_columns:
                pos_count = (split_df[attr] == 1).sum()
                total_count = len(split_df)
                stats[f'{split_name}_{attr}_positive_ratio'] = pos_count / total_count
        
        # Save stats to file
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(self.processed_path, 'dataset_stats.csv'), index=False)
        
        logger.info("Summary statistics saved to dataset_stats.csv")
        
        return stats
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting CelebA dataset preprocessing...")
        
        # Step 1: Load metadata
        full_df = self.load_metadata()
        
        # Step 2: Verify images exist
        full_df = self.verify_images_exist(full_df)
        
        # Step 3: Sample 50% of the dataset
        sampled_df = self.sample_dataset(full_df)
        
        # Step 4: Create train/test/validation splits
        train_df, test_df, val_df = self.create_splits(sampled_df)
        
        # Step 5: Save splits
        metadata_df = self.save_splits(train_df, test_df, val_df, sampled_df)
        
        # Step 6: Generate summary statistics
        stats = self.generate_summary_stats(train_df, test_df, val_df)
        
        logger.info("=== Preprocessing Complete ===")
        logger.info(f"Total processed samples: {stats['total_samples']}")
        logger.info(f"Train: {stats['train_samples']} ({stats['train_ratio']*100:.1f}%)")
        logger.info(f"Test: {stats['test_samples']} ({stats['test_ratio']*100:.1f}%)")
        logger.info(f"Validation: {stats['val_samples']} ({stats['val_ratio']*100:.1f}%)")
        
        return {
            'train_df': train_df,
            'test_df': test_df,
            'val_df': val_df,
            'metadata_df': metadata_df,
            'stats': stats
        }

def main():
    """Main function to run preprocessing"""
    # Set paths
    data_path = r"c:\Users\deepa rajesh\OneDrive\Desktop\faceauth\data"
    
    # Initialize preprocessor
    preprocessor = CelebAPreprocessor(data_path)
    
    # Run preprocessing
    results = preprocessor.run_preprocessing()
    
    print("\n" + "="*50)
    print("CelebA Dataset Preprocessing Complete!")
    print("="*50)
    print(f"Processed files saved to: {preprocessor.processed_path}")
    print("\nGenerated files:")
    print("- train.csv: Training set")
    print("- test.csv: Test set") 
    print("- val.csv: Validation set")
    print("- metadata.csv: Combined metadata")
    print("- dataset_stats.csv: Summary statistics")
    
    return results

if __name__ == "__main__":
    results = main()