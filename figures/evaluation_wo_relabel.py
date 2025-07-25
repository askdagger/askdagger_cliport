import os
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize
import numpy as np
from matplotlib import pyplot as plt

from askdagger_cliport.utils import utils

if __name__ == "__main__":
    utils.set_plot_style()
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algos = {
        "ASkDAgger": cmap[0],
        "ASkDAgger w/o relabel": cmap[8],
    }
    interactive_demos = 300
    tasks = [
        "packing-seen-google-objects-seq",
        "packing-seen-google-objects-group",
        "packing-seen-shapes",
        "put-block-in-bowl-seen-colors",
    ]
    fig, ax = plt.subplots(2, 4, figsize=(6.50127, 0.6 * 6.50127))
    offset = -30
    for askdagger in [False, True]:
        relabeling_demos = askdagger
        validation_demos = True
        pier = True
        exp_folders = ["exps"] if askdagger else ["exps_wo_relabel"]
        for exp_folder in exp_folders:
            offset += 20
            overrides_dict={
                "interactive_demos": interactive_demos,
                "relabeling_demos": relabeling_demos,
                "validation_demos": validation_demos,
                "eval.pier": pier,
                "exp_folder": exp_folder,
            }
            ax_idx = -1
            setting = f"rd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}_f={exp_folder}"
            for seen in [True, False]:
                all_task_results = []
                for task in tasks:
                    ax_idx += 1
                    eval_task = task
                    eval_task = eval_task.replace("seen", "unseen") if not seen else eval_task
                    results = np.zeros((3, 10))
                    title_split = eval_task.split("-")
                    title = title_split[0]
                    for i, split in enumerate(title_split[1:]):
                        if (i + 1) % 3 == 0:
                            title += f"\n{split}"
                        else:
                            title += f"-{split}"
                    ax[ax_idx // 4, ax_idx % 4].set_title(title)
                    
                    if ax_idx == 0 and seen:
                        if askdagger:
                            algo = "ASkDAgger"
                        elif exp_folder == "exps_wo_relabel":
                            algo = "ASkDAgger w/o relabel"
                        else:
                            raise ValueError(f"Unknown exp_folder: {exp_folders}")
                        color = algos[algo]

                    if ax_idx % 4 == 0:
                        if seen:
                            ax[ax_idx // 4, ax_idx % 4].set_ylabel(r"\textbf{Seen:} Reward")
                        else:
                            ax[ax_idx // 4, ax_idx % 4].set_ylabel(r"\textbf{Unseen:} Reward")
                    if ax_idx // 2 >= 2:
                        ax[ax_idx // 4, ax_idx % 4].set_xlabel("Demonstrations")
                    
                    for iteration in range(10):
                        overrides_dict["iteration"] = iteration
                        overrides_dict["model_task"] = task
                        overrides_dict["eval_task"] = eval_task
                        overrides = [f"{key}={value}" for key, value in overrides_dict.items()]
                        hydra.core.global_hydra.GlobalHydra.instance().clear()
                        os.chdir(os.environ["ASKDAGGER_ROOT"])
                        with initialize(config_path="../src/askdagger_cliport/cfg"):
                            vcfg = compose(
                                config_name="eval",
                                overrides=overrides,
                            )
                        OmegaConf.set_struct(vcfg, False)
                        eval_results = utils.get_eval_results(vcfg)
                        if eval_results is not None:
                            for i, interactive_demos in enumerate([100, 200, 300]):
                                ckpt = f"interactive={interactive_demos}.ckpt"
                                if ckpt in eval_results:
                                    results[i, iteration] = eval_results[ckpt]["mean_reward"] * 100
                                else:
                                    print(f"No results for checkpoint {ckpt}")
                                    results[i, iteration] = np.nan
                                    break
                    for i, interactive_demos in enumerate([100, 200, 300]):
                        if np.isnan(np.sum(results[i])):
                            continue
                        label = algo if ax_idx == 0 and i == 0 else None
                        ax[ax_idx // 4, ax_idx % 4].bar(
                            interactive_demos + offset,
                            np.mean(results[i]),
                            yerr=np.std(results[i]),
                            label=label,
                            width=20,
                            color=color,
                        )
                    all_task_results.extend(list(results.flatten()))
                print(np.mean(all_task_results))

        fig.tight_layout(rect=[0, 0.05, 1, 1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    labels = [
        r"\textbf{A}",
        r"\textbf{B}",
        r"\textbf{C}",
        r"\textbf{D}",
        r"\textbf{E}",
        r"\textbf{F}",
        r"\textbf{G}",
        r"\textbf{H}",
    ]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
        axis.set_xticks([100, 200, 300])
    plt.savefig("figures/evaluation_wo_relabel.pdf")
