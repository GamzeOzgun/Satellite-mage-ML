import rasterio                             # Import rasterio
from rasterio.plot import show              # Import show from rasterio.plot
from matplotlib import pyplot as plt        # Import pyplot from matplotlib as plt

# Open the image file
img = rasterio.open('satellite_opencv.tif')       # Open the image
show(img)          # Show the image

# Check metadata to confirm X and Y are latitude and longitude

# Read the full image including all bands
full_img = img.read()    # Note the number and shape of the bands

# Find the number of bands in the image
num_bands = img.count               # Number of bands
print("Number of bands in the image = ", num_bands)     # Print the number of bands

# Read individual bands
img_band1 = img.read(1)       # Band 1
img_band2 = img.read(2)       # Band 2
img_band3 = img.read(3)       # Band 3

# Plot the individual bands
fig = plt.figure(figsize=(10,10))      # Create a figure with size (10, 10)
ax1 = fig.add_subplot(2,2,1)           # Add a subplot for Band 1
ax1.imshow(img_band1, cmap='pink')     # Display Band 1 with a pink colormap
ax2 = fig.add_subplot(2,2,2)           # Add a subplot for Band 2
ax2.imshow(img_band2, cmap='pink')     # Display Band 2 with a pink colormap
ax3 = fig.add_subplot(2,2,3)           # Add a subplot for Band 3
ax3.imshow(img_band3, cmap='pink')     # Display Band 3 with a pink colormap

# Learn about the coordinate reference system
print("Coordinate reference system: ", img.crs)       # Print the coordinate reference system

# Read and print metadata
metadata = img.meta         # Read metadata
print('Metadata: {metadata}\n'.format(metadata=metadata))       # Print metadata

# Read and print descriptions if available
desc = img.descriptions         # Read descriptions
print('Raster description: {desc}\n'.format(desc=desc))       # Print descriptions

# Learn about the geotransform
print("Geotransform : ", img.transform)         # Print the geotransform

# Plot histograms for pixel values in each band
rasterio.plot.show_hist(full_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)
# Peak at 255 indicates pixels with no data, outside the region of interest

# Clip a region from the image and plot
clipped_img = full_img[:, 300:900, 300:900]
plt.imshow(clipped_img[0,:,:])
rasterio.plot.show_hist(clipped_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)
# Peak at 255 indicates pixels with no data, outside the region of interest

################ NDVI - NORMALIZED DIFFERENCE VEGETATION INDEX ###########
# NDVI = (NIR - RED) / (NIR + RED)

# Assume Band 1 is Red and Band 2 is NIR
red_clipped = clipped_img[0].astype('f4')
nir_clipped = clipped_img[1].astype('f4')
ndvi_clipped = (nir_clipped - red_clipped) / (nir_clipped + red_clipped)

# To handle runtime warnings due to division by zero, use numpy and replace inf / nan with a value
import numpy as np          # Import numpy as np

ndvi_clipped2 = np.divide(np.subtract(nir_clipped, red_clipped), np.add(nir_clipped, red_clipped))
# Compute NDVI using numpy

ndvi_clipped3 = np.nan_to_num(ndvi_clipped2, nan=-1)
# Replace NaNs with -1

plt.imshow(ndvi_clipped3, cmap='viridis')
# Display NDVI with a viridis colormap

plt.colorbar()     # Add a colorbar

# Sometimes bands are available as separate images
# Data from: !!!"""https://landsatonaws.com/L8/042/034/LC08_L1TP_042034_20180619_20180703_01_T1"""!!
# Band 4 = Red, Band 5 = NIR

# Open Red band image
red = rasterio.open('satellite_opencv.tif')        # Open the Red band image

# Read the image at a smaller size
red_img = red.read(1, out_shape=(1, int(red.height // 2), int(red.width // 2)))
# Read Red band image at half size

plt.imshow(red_img, cmap='viridis')
plt.colorbar()    # Display image and colorbar

# Extract a smaller region to avoid division by zero in NDVI calculation
red_img = red_img[500:1500, 500:1500]
plt.imshow(red_img, cmap='viridis')
plt.colorbar()

# Open NIR band image
nir = rasterio.open('satellite_opencv.tif')  # Open the NIR band image
nir_img = nir.read(1, out_shape=(1, int(nir.height // 2), int(nir.width // 2)))
# Read NIR band image at half size
nir_img = nir_img[500:1500, 500:1500]

plt.imshow(nir_img, cmap='viridis')
plt.colorbar()

# Convert images to float for NDVI calculation
red_img_float = red_img.astype('f4') # Float 32
nir_img_float = nir_img.astype('f4')

# Calculate NDVI
ndvi = (nir_img_float - red_img_float) / (nir_img_float + red_img_float)
plt.imshow(ndvi, cmap='viridis')
plt.colorbar()