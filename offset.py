


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import matplotlib

# ===== 1. Load and clean LaTeX table =====
file_path = "/Users/jose/FL_IoT_Network/offset.tex"
rows = []
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith("\\") or line == "" or line.startswith("\\hline") or line.startswith("\\midrule"):
            continue
        parts = [col.strip() for col in line.strip("\\").split("&")]
        rows.append(parts)

header = rows[0]
data = rows[1:]
df = pd.DataFrame(data, columns=header)
df.columns = df.columns.str.strip()

# Convert numeric columns
numeric_cols = ['Alpha', 'S-LR', 'FL Param', 'Accuracy', 'F1-Weighted', 'MCS', 'Loss', 'Runtime (s)']
df[numeric_cols] = df[numeric_cols].replace(['-', 'None', 'none', 'NULL', 'null'], np.nan)
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ===== 2. Filter out rows with Alpha <= 0 to avoid negative / zero values on log scale =====
df = df[df['Alpha'] > 0]

# ===== 3. Setup plot parameters =====
metrics = ['Accuracy', 'F1-Weighted', 'MCS', 'Loss', 'Runtime (s)']
selected_methods = ['FedAvg', 'FedAvgM', 'FedOpt', 'QFedAvg']  # Added QFedAvg


def create_latex_legend():
    # Generate the summary text in the exact same format as before
    summary_text = "Metrics Legend:\n" \
                   "• Accuracy: Classification accuracy (0-1)\n" \
                   "• F1-Weighted: F1 score weighted by support\n" \
                   "• MCS: Model confidence score (0-1)\n" \
                   "• Loss: Training loss (lower better)\n" \
                   "• Runtime: Execution time in seconds\n\n" \
                   "Parameters Legend:\n" \
                   "• α (X-axis): Global learning rate (log scale)\n" \
                   "• S-LR (Y-axis): Server learning rate\n" \
                   "• FL Param (Z-axis): Client parameter\n\n" \
                   "Best Performance Summary:\n\n"

    for method in selected_methods:
        summary_text += f"{method}:\n"
        for metric in metrics:
            subset = df[df['FL'] == method][metric].dropna()
            if not subset.empty:
                if metric in ['Loss', 'Runtime (s)']:
                    best_val = subset.min()
                    summary_text += f"• {metric}: {best_val:.3f} (min)\n"
                else:
                    best_val = subset.max()
                    summary_text += f"• {metric}: {best_val:.3f} (max)\n"
        summary_text += "\n"

    # Convert to LaTeX format with proper escaping
    latex_text = summary_text.replace('•', r'$\bullet$').replace('\n', '\\\\\n').replace('α', r'$\alpha$')
    
    latex_code = fr"""
    \documentclass[border=5pt]{{standalone}}
    \usepackage{{amsmath}}
    \usepackage{{amssymb}}
    \usepackage{{xcolor}}
    \usepackage{{array}}
    \usepackage{{booktabs}}
    
    \begin{{document}}
    \begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{5cm}}l}}
    \toprule
    \textbf{{Federated Learning Results Summary}} & \\
    \midrule
    {latex_text}
    \bottomrule
    \end{{tabular}}
    \end{{document}}
    """
    
    with open("legend_offset.tex", "w") as f:
        f.write(latex_code)
    print("LaTeX legend with original summary format saved to legend.tex")

# ===== 5. Plot styling =====
PLOT_STYLE = {
    '2d': {
        'scatter_size': 60,
        'cmap': 'viridis',
        'alpha': 0.8,
        'trendline_style': {'color': 'red', 'linestyle': '--', 'linewidth': 1}
    },
    '3d': {
        'scatter_size': 50,
        'cmap': 'viridis',
        'alpha': 0.8,
        'view_angle': {'elev': 25, 'azim': 45}
    },
    'fonts': {
        'axis_label': 10,
        'tick_label': 8,
        'title': 12,
        'method_label': 10
    }
}

# ===== 6. Create separate plots for each method =====
# for row, method in enumerate(selected_methods):
#     # Create figure for this method
#     fig = plt.figure(figsize=(20, 5), dpi=100)
    
#     # Adjust grid layout with reduced top margin
#     is_2d = (method == 'FedAvg')
#     ncols = len(metrics)
#     grid = plt.GridSpec(
#         nrows=1, ncols=ncols,
#         hspace=0.3, wspace=0.4,
#         top=0.92,  # Reduced from 0.85 to bring title closer (smaller number = closer)
#         bottom=0.15,
#         left=0.08, 
#         right=0.92
#     )
    
#     # Create subplots for each metric
#     for col, metric in enumerate(metrics):
#         ax = fig.add_subplot(
#             grid[0, col],
#             projection=None if is_2d else '3d'
#         )
        
#         # Filter data per method and remove NaNs for current metric
#         data = df[(df['FL'] == method) & df[metric].notna()]
        
#         if data.empty:
#             ax.axis('off')
#             continue
        
#         if is_2d:
#             # 2D scatter plot: x=log10(Alpha), y=metric value
#             x = np.log10(data['Alpha'])
#             y = data[metric]
            
#             sc = ax.scatter(
#                 x, y,
#                 c=y,
#                 s=PLOT_STYLE['2d']['scatter_size'],
#                 cmap=PLOT_STYLE['2d']['cmap'],
#                 alpha=PLOT_STYLE['2d']['alpha']
#             )
            
#             ax.set_xlabel('α', fontsize=PLOT_STYLE['fonts']['axis_label'])
#             ax.set_ylabel(metric, fontsize=PLOT_STYLE['fonts']['axis_label'])
            
#         else:
#             # 3D scatter plot
#             sc = ax.scatter3D(
#                 np.log10(data['Alpha']),
#                 data['S-LR'],
#                 data['FL Param'],
#                 c=data[metric],
#                 s=PLOT_STYLE['3d']['scatter_size'],
#                 cmap=PLOT_STYLE['3d']['cmap'],
#                 alpha=PLOT_STYLE['3d']['alpha']
#             )
            
#             ax.set_xlabel('α', fontsize=PLOT_STYLE['fonts']['axis_label'])
#             ax.set_ylabel('S-LR', fontsize=PLOT_STYLE['fonts']['axis_label'])
#             ax.set_zlabel('FL Param', fontsize=PLOT_STYLE['fonts']['axis_label'])
#             ax.view_init(**PLOT_STYLE['3d']['view_angle'])
        
#         # Style ticks and grid
#         ax.tick_params(axis='both', labelsize=PLOT_STYLE['fonts']['tick_label'])
#         ax.grid(True, alpha=0.3)
#         ax.set_title(metric, pad=10, fontsize=PLOT_STYLE['fonts']['title'])
        
#         # Add colorbar for each plot
#         cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
#         cbar.set_label(metric, fontsize=PLOT_STYLE['fonts']['axis_label'])
#         cbar.ax.tick_params(labelsize=PLOT_STYLE['fonts']['tick_label'])
    
  
#     # Save this method's figure
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the top parameter (0.95) to reduce space
#     output_file = f"FL_Dashboard_{method}_offset.png"
#     plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
#     print(f"Saved {output_file}")

# Calculate global min/max values for each axis (before the plotting loop)
alpha_min, alpha_max = np.log10(df['Alpha'].min()), np.log10(df['Alpha'].max())
s_lr_min, s_lr_max = df['S-LR'].min(), df['S-LR'].max()
fl_param_min, fl_param_max = df['FL Param'].min(), df['FL Param'].max()

# Calculate metric-specific ranges with 5% padding
metric_ranges = {}
for metric in metrics:
    valid_values = df[metric].dropna()
    if len(valid_values) > 0:
        range_min = valid_values.min()
        range_max = valid_values.max()
        padding = (range_max - range_min) * 0.05  # 5% padding
        metric_ranges[metric] = (max(0, range_min - padding), range_max + padding)

for row, method in enumerate(selected_methods):
    # Create figure for this method
    fig = plt.figure(figsize=(20, 5), dpi=100)
    
    # Adjust grid layout with reduced top margin
    is_2d = (method == 'FedAvg')
    ncols = len(metrics)
    grid = plt.GridSpec(
        nrows=1, ncols=ncols,
        hspace=0.3, wspace=0.4,
        top=0.92,
        bottom=0.15,
        left=0.08, 
        right=0.92
    )
    
    # Create subplots for each metric
    for col, metric in enumerate(metrics):
        ax = fig.add_subplot(
            grid[0, col],
            projection=None if is_2d else '3d'
        )
        
        # Filter data per method and remove NaNs for current metric
        data = df[(df['FL'] == method) & df[metric].notna()]
        
        if data.empty:
            ax.axis('off')
            continue
        
        if is_2d:
            # 2D scatter plot: x=log10(Alpha), y=metric value
            x = np.log10(data['Alpha'])
            y = data[metric]
            
            sc = ax.scatter(
                x, y,
                c=y,
                s=PLOT_STYLE['2d']['scatter_size'],
                cmap=PLOT_STYLE['2d']['cmap'],
                alpha=PLOT_STYLE['2d']['alpha']
            )
            
            # Set axis limits based on data ranges
            ax.set_xlim(alpha_min, alpha_max)
            if metric in metric_ranges:
                ax.set_ylim(metric_ranges[metric])
            
            # Format x-axis ticks
            x_ticks = np.linspace(alpha_min, alpha_max, 4)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'$10^{{{x:.0f}}}$' for x in x_ticks])
            
            ax.set_xlabel('α', fontsize=PLOT_STYLE['fonts']['axis_label'])
            ax.set_ylabel(metric, fontsize=PLOT_STYLE['fonts']['axis_label'])
            
        else:
            # 3D scatter plot
            sc = ax.scatter3D(
                np.log10(data['Alpha']),
                data['S-LR'],
                data['FL Param'],
                c=data[metric],
                s=PLOT_STYLE['3d']['scatter_size'],
                cmap=PLOT_STYLE['3d']['cmap'],
                alpha=PLOT_STYLE['3d']['alpha']
            )
            
            # Set axis limits based on data ranges
            ax.set_xlim(alpha_min, alpha_max)
            ax.set_ylim(s_lr_min, s_lr_max)
            ax.set_zlim(fl_param_min, fl_param_max)
            
            # Format ticks consistently
            x_ticks = np.linspace(alpha_min, alpha_max, 4)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'$10^{{{x:.0f}}}$' for x in x_ticks])
            
            y_ticks = np.linspace(s_lr_min, s_lr_max, 5)
            ax.set_yticks(y_ticks)
            
            z_ticks = np.linspace(fl_param_min, fl_param_max, 4)
            ax.set_zticks(z_ticks)
            
            ax.set_xlabel('α', fontsize=PLOT_STYLE['fonts']['axis_label'])
            ax.set_ylabel('S-LR', fontsize=PLOT_STYLE['fonts']['axis_label'])
            ax.set_zlabel('FL Param', fontsize=PLOT_STYLE['fonts']['axis_label'])
            ax.view_init(**PLOT_STYLE['3d']['view_angle'])
        
        # Style ticks and grid
        ax.tick_params(axis='both', labelsize=PLOT_STYLE['fonts']['tick_label'])
        ax.grid(True, alpha=0.3)
        ax.set_title(metric, pad=10, fontsize=PLOT_STYLE['fonts']['title'])
        
        # Add colorbar with consistent scaling
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(metric, fontsize=PLOT_STYLE['fonts']['axis_label'])
        cbar.ax.tick_params(labelsize=PLOT_STYLE['fonts']['tick_label'])
        
        # Set colorbar limits to match y-axis
        if metric in metric_ranges:
            sc.set_clim(metric_ranges[metric])
    
    # Save this method's figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = f"FL_Dashboard_{method}_offset.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {output_file}")

create_latex_legend()
print("All images generated successfully.")