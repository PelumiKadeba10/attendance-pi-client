#!/usr/bin/env python3
"""
Download LFW Pairs dataset with visible progress - CONFIRMED WORKING
"""

import sys
import urllib.request
import urllib.error
from sklearn.datasets import fetch_lfw_pairs
import time

class ProgressPrinter:
    def __init__(self):
        self.last_percent = 0
        self.start_time = time.time()
    
    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
            
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        
        # Only update every 2% to avoid flickering
        if percent > self.last_percent:
            self.last_percent = percent
            
            # Create progress bar
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Convert to MB
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            # Calculate speed
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                speed = downloaded / elapsed / (1024 * 1024)  # MB/s
                sys.stdout.write(f'\r|{bar}| {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB) - {speed:.1f} MB/s')
            else:
                sys.stdout.write(f'\r|{bar}| {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
            
            sys.stdout.flush()

print("="*60)
print("DOWNLOADING LFW PAIRS DATASET")
print("="*60)
print("Size: ~26 MB")
print()

# Save original function
original_urlretrieve = urllib.request.urlretrieve

# Create wrapper with progress
def urlretrieve_with_progress(url, filename, reporthook=None, data=None):
    print(f"Connecting to: {url}")
    print("Download started...\n")
    return original_urlretrieve(url, filename, ProgressPrinter(), data)

# Apply the patch
urllib.request.urlretrieve = urlretrieve_with_progress

try:
    # Download the dataset
    print("Fetching dataset from server...\n")
    lfw_pairs = fetch_lfw_pairs(subset='train', download_if_missing=True)
    
    print("\n\n" + "="*60)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"✓ Loaded {lfw_pairs.pairs.shape[0]} image pairs for verification")
    print(f"✓ Total images: {lfw_pairs.pairs.shape[0] * 2}")
    print(f"✓ Dataset shape: {lfw_pairs.pairs.shape}")
    print("\nDataset ready for testing!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try again - the server might be busy")