from utils.robustness_evaluation import robustness_evaluation, robustness_evaluation_plot_data, venn_diagam_data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from venn import venn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np


out_dir_liver = "experiments/reproduction/outputs/liverfailure"
root_paths_liver = [f"{out_dir_liver}_{i}/causality_output/root.csv" for i in range(1, 4)]
out_dir_tramadol = "experiments/reproduction/outputs/tramadol_corrected"
root_paths_tramadol = [f"{out_dir_tramadol}_{i}/causality_output/root.csv" for i in range(1, 4)]


for path in root_paths_liver:
    print(path)
for path in root_paths_tramadol:
    print(path)

x_axis_liver, y_axis_liver = robustness_evaluation_plot_data(root_paths_liver)
res_liver = venn_diagam_data(root_paths_liver)
x_axis_tramadol, y_axis_tramadol = robustness_evaluation_plot_data(root_paths_tramadol)
res_tramadol = venn_diagam_data(root_paths_tramadol)

# Shades of blue cmap


# Load the original "Blues" colormap
cmap_blues = plt.cm.Blues
cmap_oranges = plt.cm.Oranges

# Create a custom colormap that excludes the lighter part
# For example, use only the colors from 0.3 to 1.0 of the original colormap
start = 0.3
stop = 1.0
colors_blues = cmap_blues(np.linspace(start, stop, cmap_blues.N))
custom_cmap_blues = mcolors.LinearSegmentedColormap.from_list('custom_blues', colors_blues)
colors_oranges = cmap_oranges(np.linspace(start, stop, cmap_oranges.N))
custom_cmap_oranges = mcolors.LinearSegmentedColormap.from_list('custom_oranges', colors_oranges)


# Create your main plot
plt.plot(x_axis_tramadol, y_axis_tramadol, marker='o', linestyle='--', label='Tramadol (Corrected)', color=custom_cmap_oranges(0.6), alpha=0.6)
plt.plot(x_axis_liver, y_axis_liver, marker='o', linestyle='--', label='Liverfailure', color=custom_cmap_blues(0.6),alpha=0.8)
plt.xlabel('Number of enhanced terms')
plt.gca().set_xticks(plt.gca().get_xticks()[::2])
plt.ylim(0, 1)
plt.ylabel('Percentage of overlapped terms')
plt.legend(fontsize=9, loc="lower left", ncol = 2)
# grid 
plt.title('Robustness evaluation')

inset1_pos = [0.5, 0.1, 0.4, 0.4]  # [x, y, width, height] for the first inset
inset2_pos = [0.1, 0.1, 0.4, 0.4]  # [x, y, width, height] for the second inset

# Create an inset for the Venn diagram
# Adjust the parameters of inset_axes to position and scale your Venn diagram
# ax_inset = inset_axes(plt.gca(), width="40%", height="60%", loc='lower right')
gca = plt.gca()

ax_inset1 = inset_axes(gca, width="100%", height="110%", loc='lower right', bbox_to_anchor=inset1_pos, bbox_transform=gca.transAxes)
# Now plot the Venn diagram in the inset
venn(res_liver, cmap=custom_cmap_blues, fontsize=9, legend_loc="lower right", ax=ax_inset1)

# ax_inset = inset_axes(plt.gca(), width="40%", height="60%", loc='lower left')
ax_inset2 = inset_axes(gca, width="100%", height="110%", loc='lower right', bbox_to_anchor=inset2_pos, bbox_transform=gca.transAxes)
# Now plot the Venn diagram in the inset
venn(res_tramadol, cmap=custom_cmap_oranges, fontsize=9, legend_loc="lower right", ax=ax_inset2)

# save to pdf
plt.draw()
plt.savefig(f'plots/robustness_evaluation.pdf', format='pdf', dpi=300)


plt.show()