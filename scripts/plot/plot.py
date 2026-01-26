import pandas as pd
import matplotlib.pyplot as plt
import io

# Data with 1.5s runtime for Ours
csv_data = """ratio,method,psnr,time,memory
10,GHAP,23.279,5.316,4.719
10,LightGaussian,22.079,10.516,3.792
10,MesonGS,20.681,10.716,2.739
10,PUP-3DGS,21.502,26.017,3.883
10,Trimming the Fat,21.502,0.000,3.628
20,GHAP,23.582,4.117,4.756
20,LightGaussian,22.453,10.916,3.792
20,MesonGS,20.666,10.816,2.739
20,PUP-3DGS,22.569,26.017,3.883
20,Trimming the Fat,22.465,0.000,3.619
30,GHAP,23.706,3.516,4.835
30,LightGaussian,22.548,8.717,3.792
30,MesonGS,20.857,10.516,2.739
30,PUP-3DGS,23.069,26.017,3.883
30,Trimming the Fat,22.768,0.000,3.610
40,GHAP,23.623,3.016,4.917
40,LightGaussian,22.555,10.415,3.792
40,MesonGS,21.108,10.616,2.739
40,PUP-3DGS,23.360,26.017,3.883
40,Trimming the Fat,22.858,0.000,3.647
50,GHAP,23.757,2.717,4.996
50,LightGaussian,22.576,10.516,3.792
50,MesonGS,21.398,10.816,2.739
50,PUP-3DGS,23.549,26.017,3.883
50,Trimming the Fat,22.878,0.000,3.687
10,Ours,23.420,1.500,3.650
20,Ours,23.69,1.500,3.650
30,Ours,23.80,1.500,3.650
40,Ours,23.870,1.500,3.650
50,Ours,23.950,1.500,3.650"""

df = pd.read_csv(io.StringIO(csv_data))

# --- ICML Style Configuration ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# Colors & Markers
colors = {
    'Ours': '#d62728',       # Bold Red
    'GHAP': '#1f77b4',       # Standard Blue
    'LightGaussian': '#aec7e8', # Light Blue
    'PUP-3DGS': '#ffbb78',   # Light Orange
    'Trimming the Fat': '#9467bd', # Purple
    'MesonGS': '#c5b0d5'     # Light Purple
}
markers = {
    'Ours': '*', 'GHAP': 'o', 'LightGaussian': 's',
    'PUP-3DGS': 'D', 'Trimming the Fat': '^', 'MesonGS': 'v'
}
linewidths = {
    'Ours': 4.0, 'GHAP': 3.0, 'LightGaussian': 2.0,
    'PUP-3DGS': 2.0, 'Trimming the Fat': 2.0, 'MesonGS': 2.0
}
markersizes = {
    'Ours': 12, 'GHAP': 9, 'LightGaussian': 7,
    'PUP-3DGS': 7, 'Trimming the Fat': 7, 'MesonGS': 7
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = [
    ('psnr', 'PSNR (dB) ↑', 'Rendering Quality'),
    ('time', 'Time (s) ↓', 'Compaction Time'),
    ('memory', 'Memory (GB) ↓', 'Peak Memory')
]

for i, (metric, label, title) in enumerate(metrics):
    ax = axes[i]
    for method in df['method'].unique():
        data = df[df['method'] == method]
        ax.plot(data['ratio'], data[metric],
                label=method,
                color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'),
                linewidth=linewidths.get(method, 1.5),
                markersize=markersizes.get(method, 6))

    # X-axis
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xlabel('Retention Ratio (%)', fontweight='bold')

    # Y-axis
    if metric == 'psnr':
        # Strictly Integer Ticks
        ax.set_yticks([21, 22, 23, 24])
        # Slightly wider limits to show all data points without clipping
        ax.set_ylim(20.5, 24.5)
    else:
        yl = ax.get_ylim()
        y_range = yl[1] - yl[0]
        ax.set_ylim(yl[0] - y_range*0.05, yl[1] + y_range*0.05)

    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)

# Legend in Plot 1
axes[0].legend(loc='lower right', framealpha=0.9, edgecolor='gray', fancybox=False)

plt.tight_layout()
plt.savefig('final_icml_plot_integer_ticks.pdf', dpi=300, bbox_inches='tight')
plt.show()