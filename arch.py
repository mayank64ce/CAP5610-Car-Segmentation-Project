import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

# Helper function to add a box with text
def add_box(ax, xy, width, height, text, color, fontsize=10):
    rect = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.1", edgecolor=color, facecolor="white", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=color)

# Helper function to add an arrow
def add_arrow(ax, start, end, color="black"):
    arrow = Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=0.5, edgecolor=color, facecolor=color)
    ax.add_patch(arrow)

# Create the figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 20)
ax.set_ylim(0, 25)
ax.axis("off")

# Vision Transformer backbone
add_box(ax, (6, 18), 8, 2, "Vision Transformer (ViT)", "blue")

# MLP + Segmentation Upscaler pathway
add_box(ax, (3, 12), 6, 2, "MLP Head", "green")
add_arrow(ax, (6, 18), (6, 14))
add_box(ax, (3, 9), 6, 2, "Grid Reshape", "purple")
add_arrow(ax, (6, 12), (6, 10))
add_box(ax, (3, 6), 6, 2, "Segmentation Upscaler\n(ConvTranspose2D)", "orange")
add_arrow(ax, (6, 9), (6, 7))
add_box(ax, (3, 3), 6, 2, "Output Mask", "red")
add_arrow(ax, (6, 6), (6, 4))

# Conv Upscaler pathway
add_box(ax, (11, 12), 6, 2, "Grid Reshape", "purple")
add_arrow(ax, (10, 18), (14, 14))
add_box(ax, (11, 9), 6, 2, "Conv2D + ReLU", "green")
add_arrow(ax, (14, 12), (14, 10))
add_box(ax, (11, 6), 6, 2, "Upsampling\n(Bilinear Interpolation)", "orange")
add_arrow(ax, (14, 9), (14, 7))
add_box(ax, (11, 3), 6, 2, "Output Mask", "red")
add_arrow(ax, (14, 6), (14, 4))

# Labels for pathways
ax.text(6, 22, "Shared Backbone: Vision Transformer", ha="center", fontsize=12, fontweight="bold", color="blue")
ax.text(6, 2, "MLP + Segmentation Upscaler Pathway", ha="center", fontsize=10, fontweight="bold", color="black")
ax.text(14, 2, "Conv Upscaler Pathway", ha="center", fontsize=10, fontweight="bold", color="black")

plt.tight_layout()
plt.show()
plt.savefig('arch.png')