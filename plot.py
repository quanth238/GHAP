import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Define the color map (Yellow-Orange-Brown matches your image best)
cmap = mpl.cm.YlOrBr

# Create the colorbar base
norm = mpl.colors.Normalize(vmin=0, vmax=100)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')

# Labeling the bar according to your "Weight Focus" description
cb1.set_label('Focus Weight Intensity')
cb1.set_ticks([0, 50, 100])
cb1.set_ticklabels(['Low Focus\n(Background)', 'Medium Focus', 'High Focus\n(The Eyes)'])

plt.title("Heatmap Scale")
plt.show()