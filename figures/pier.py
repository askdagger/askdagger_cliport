import numpy as np
from matplotlib import pyplot as plt
from askdagger_cliport.utils import utils


if __name__ == "__main__":
    utils.set_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(6.50127 * 0.5, 6.50127 * 0.5))

    c = np.linspace(0, 1, 100)
    for b, linestyle in zip([1.001, 10, 1000], ["-", "-.", "--"]):
        for r, color in zip([-1, 0, 1], ["r", "k", "g"]):
            p = 1 - r * (b ** (1 - c) - 1) / (b - 1)
            ax.plot(c, p, color=color, linestyle=linestyle)
            if r == 1:
                ax.text(c[len(c) // 3], p[len(c) // 3], rf"b={b}", color=color, fontsize=8, verticalalignment="top")
            elif r == -1:
                ax.text(c[len(c) // 3], p[len(c) // 3], rf"b={b}", color=color, fontsize=8, verticalalignment="bottom")
    ax.set_xlabel(r"Prioritization Exponent $c$")
    ax.set_ylabel(r"Priorities $p$")
    handles = [
        plt.Line2D([0], [0], color="r", label="r = -1"),
        plt.Line2D([0], [0], color="k", label="r = 0"),
        plt.Line2D([0], [0], color="g", label="r = 1"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig("figures/pier.pdf")
