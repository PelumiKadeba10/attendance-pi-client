import sys
from sklearn.datasets import fetch_lfw_pairs

# Custom progress printer
class ProgressPrinter:
    def __init__(self):
        self.last_percent = 0
    
    def __call__(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        
        if percent > self.last_percent:
            self.last_percent = percent
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Convert to MB for display
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            sys.stdout.write(f'\rDownloading: |{bar}| {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()

print('Downloading LFW Pairs dataset (approx. 26 MB)...')
print()

# Monkey patch urllib to show progress
import urllib.request
original_urlretrieve = urllib.request.urlretrieve

def urlretrieve_with_progress(url, filename, reporthook=None, data=None):
    return original_urlretrieve(url, filename, ProgressPrinter(), data)

urllib.request.urlretrieve = urlretrieve_with_progress

# Download the dataset
lfw_pairs = fetch_lfw_pairs(subset='train', download_if_missing=True)

print('\n\n✓ Download complete!')
print(f'Success! Loaded {lfw_pairs.pairs.shape[0]} image pairs for verification.')
