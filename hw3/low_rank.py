# ECS130 HW3 -- Image Compression via Low-Rank Approximation
import sys
from matplotlib import pyplot as plt
import svd_solution as svd
import numpy as np

# Parse command line arguments.
# Example usage: python low_rank.py 10 in.png out.png
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('rank', type=int)
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()
# Load the input image and convert to double precision.
img = np.array(plt.imread(args.input), dtype=np.float64)

# Compute the SVD.
U, sigma, V = svd.svd(img)

# Truncate the SVD matrices to the given rank.
r = args.rank
U = U[:, :r]
sigma = sigma[:r]
V = V[:, :r]

# Reconstruct the image using the truncated SVD.
img_reconstructed = U @ np.diag(sigma) @ V.T

# Save the reconstructed image.
plt.imsave(args.output, img_reconstructed, cmap='gray')

# Print the compression ratio.
print(f'Compression ratio: {img.size / (U.size + sigma.size + V.size):.3}x')
