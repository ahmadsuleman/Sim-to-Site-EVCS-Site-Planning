import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Oracle sensitivity data
# Mean assigned distance excluded
# -----------------------------
data = [
    ["Station budget", "K=5", 0.660, 0.602, 0.657, 0.292, 8.404],
    ["Station budget", "K=10", 0.923, 0.799, 0.686, 0.343, 42.404],
    ["Station budget", "K=20", 0.925, 0.800, 0.666, 0.339, 29.631],

    ["Service radius", "R=3 km", 0.866, 0.694, 0.682, 0.357, 43.656],
    ["Service radius", "R=5 km", 0.923, 0.799, 0.686, 0.343, 42.713],
    ["Service radius", "R=8 km", 0.990, 0.966, 0.695, 0.341, 24.609],

    ["Capacity multiplier", r"$w_{\mathrm{cap}}=0.75$", 0.881, 0.770, 0.659, 0.326, 19.825],
["Capacity multiplier", r"$w_{\mathrm{cap}}=1.00$", 0.923, 0.799, 0.686, 0.343, 41.278],
["Capacity multiplier", r"$w_{\mathrm{cap}}=1.25$", 0.934, 0.806, 0.689, 0.344, 26.500],
    ["Objective weights", r"$(w_{\mathrm{suit}},w_{\mathrm{cost}})=(0,0.25)$", 0.919, 0.796, 0.486, 0.133, 33.515],
["Objective weights", r"$(w_{\mathrm{suit}},w_{\mathrm{cost}})=(0.5,0.25)$", 0.923, 0.799, 0.686, 0.343, 42.712],
["Objective weights", r"$(w_{\mathrm{suit}},w_{\mathrm{cost}})=(1,0.25)$", 0.904, 0.797, 0.713, 0.400, 23.866],
["Objective weights", r"$(w_{\mathrm{suit}},w_{\mathrm{cost}})=(0.5,0)$", 0.923, 0.799, 0.715, 0.458, 34.896],
["Objective weights", r"$(w_{\mathrm{suit}},w_{\mathrm{cost}})=(0.5,0.5)$", 0.920, 0.791, 0.565, 0.183, 34.472],
]
# r'$(w_{\mathrm{suit}} = 0, w_{\mathrm{cost}} = 0.25)$'
    # ["Objective weights", "(β=0,μ=0.25)", 0.919, 0.796, 0.486, 0.133, 33.515],
    # ["Objective weights", "(β=0.5,μ=0.25)", 0.923, 0.799, 0.686, 0.343, 42.712],
    # ["Objective weights", "(β=1,μ=0.25)", 0.904, 0.797, 0.713, 0.400, 23.866],
    # ["Objective weights", "(β=0.5,μ=0)", 0.923, 0.799, 0.715, 0.458, 34.896],
    # ["Objective weights", "(β=0.5,μ=0.5)", 0.920, 0.791, 0.565, 0.183, 34.472],
# ]

df = pd.DataFrame(
    data,
    columns=[
        "Sensitivity group",
        "Setting",
        "Coverage rate",
        "Underserved coverage",
        "Suitability",
        "Cost proxy",
        "Runtime"
    ]
)

# -----------------------------
# Plot setup
# -----------------------------
x = np.arange(len(df))

colors = {
    "Coverage rate": "#0072B2",          # blue
    "Underserved coverage": "#009E73",   # green
    "Suitability": "#D55E00",            # orange-red
    "Cost proxy": "#CC79A7",             # purple-pink
    "Runtime": "#F0E442"                 # yellow
}

fig, ax1 = plt.subplots(figsize=(16, 7))

# Background group shading
group_boundaries = []
start = 0
for group, sub in df.groupby("Sensitivity group", sort=False):
    end = start + len(sub) - 1
    group_boundaries.append((group, start, end))
    ax1.axvspan(start - 0.5, end + 0.5, alpha=0.08)
    start = end + 1

# Main performance lines
metrics = ["Coverage rate", "Underserved coverage", "Suitability", "Cost proxy"]

for metric in metrics:
    ax1.plot(
        x,
        df[metric],
        marker="o",
        linewidth=2.8,
        markersize=7,
        label=metric,
        color=colors[metric]
    )

# Secondary axis for runtime
ax2 = ax1.twinx()
ax2.bar(
    x,
    df["Runtime"],
    width=0.55,
    alpha=0.32,
    color=colors["Runtime"],
    edgecolor="black",
    linewidth=0.6,
    label="Runtime"
)

# Vertical separators between groups
for _, _, end in group_boundaries[:-1]:
    ax1.axvline(end + 0.5, color="black", linestyle="--", linewidth=1, alpha=0.45)

# Group labels above chart
for group, start, end in group_boundaries:
    center = (start + end) / 2
    ax1.text(
        center,
        1.055,
        group,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        transform=ax1.get_xaxis_transform()
    )

# # Axis formatting
# ax1.set_title(
#     "Oracle Sensitivity Summary Across Planning Parameters",
#     fontsize=17,
#     fontweight="bold",
#     pad=28
# )

ax1.set_ylabel("Normalized Metrics Value", fontsize=18, fontweight="bold")
ax2.set_ylabel("Runtime (s)", fontsize=18, fontweight="bold")

ax1.set_ylim(0.0, 1.08)
ax2.set_ylim(0, max(df["Runtime"]) * 1.25)

ax1.set_xticks(x)
ax1.set_xticklabels(df["Setting"], rotation=35, ha="right", fontsize=16)

# Add y-axis tick size here
ax1.tick_params(axis="y", labelsize=18)
ax2.tick_params(axis="y", labelsize=18)

ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.set_axisbelow(True)

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

legend = ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.35),
    ncol=5,
    frameon=True,
    fontsize=18
)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.6)

# # Caption-style note
# fig.text(
#     0.5,
#     -0.02,
#     "Higher coverage, underserved coverage, and suitability are preferred; lower cost proxy and runtime are preferred.",
#     ha="center",
#     fontsize=14
# )

plt.tight_layout()

# Save high-resolution outputs
plt.savefig("oracle_sensitivity_summary.png", dpi=400, bbox_inches="tight")
plt.savefig("oracle_sensitivity_summary.pdf", bbox_inches="tight")

plt.show()