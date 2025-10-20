import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Read in PNG images
img1 = mpimg.imread('/u/ferhi/Figures/d_lim.png')
img2 = mpimg.imread("cloud_separation_8rcl_1000_.png")

# Create figure and GridSpec (1 row, 2 columns)
fig = plt.figure(figsize=(6,6))
gs = gridspec.GridSpec(2, 3, width_ratios = [0.05, 1, 0.1], height_ratios= [1, 1.2], figure=fig, wspace=0.05, hspace=0)  # wspace adjusts spacing

# Left subplot
ax1 = fig.add_subplot(gs[0, :])
ax1.imshow(img1)
ax1.axis("off")

# Right subplot
ax2 = fig.add_subplot(gs[1, 1])
ax2.imshow(img2)
ax2.axis("off")

plt.savefig("combined_figure.png", dpi=300, bbox_inches='tight')