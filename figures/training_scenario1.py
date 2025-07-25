import os
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize
import numpy as np
from matplotlib import pyplot as plt

import askdagger_cliport
from askdagger_cliport.utils import utils
from itertools import zip_longest


def column_wise_mean_std(rows):
    columns = zip_longest(*rows, fillvalue=np.nan)
    mean_std = np.asarray([[np.nanmean(col), np.nanstd(col)] for col in columns])
    return mean_std[:, 0], mean_std[:, 1]


if __name__ == "__main__":
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    utils.set_plot_style()
    algos = {
        "ASkDAgger": cmap[0],
        "ActiveDAgger": cmap[1],
        "ThriftyDAgger": cmap[6],
        "SafeDAgger": cmap[7],
    }
    results = {}
    interactive_demos = 150
    tasks = [
        "packing-seen-google-objects-seq",
        "packing-seen-google-objects-group",
        "packing-seen-shapes",
        "put-block-in-bowl-seen-colors",
    ]
    fig, ax = plt.subplots(1, 4, figsize=(6.50127, 0.3 * 6.50127))
    for askdagger in [False, True]:
        pier = askdagger
        relabeling_demos = askdagger
        validation_demos = askdagger
        exp_folders = ["exps"] if askdagger else ["exps", "exps_thrifty", "exps_safe"]
        for exp_folder in exp_folders:
            ax_idx = -1

            setting = f"rd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}_f={exp_folder}"
            
            overrides_dict={
                "interactive_demos": interactive_demos,
                "relabeling_demos": relabeling_demos,
                "validation_demos": validation_demos,
                "eval.pier": pier,
                "exp_folder": exp_folder,
            }
            if exp_folder == "exps_thrifty":
                overrides_dict["agent"] = "dropout_cliport"

            for task in tasks:
                ax_idx += 1
                if ax_idx % 4 == 0:
                    ax[ax_idx].set_ylabel("Novice Success Rate")
                ax[ax_idx].set_xlabel("Ann. Tuples")
                setting_results = dict(n_annotation=[], success=[])

                title_split = task.split("-")
                if "seen" in title_split:
                    title_split.remove("seen")
                if "colors" in title_split:
                    title_split.remove("colors")
                title = title_split[0]
                for i, split in enumerate(title_split[1:]):
                    if i % 2 == 0:
                        title += f"-{split}"
                    else:
                        title += f"\n{split}"
                ax[ax_idx].set_title(title)

                if ax_idx == 0:
                    if askdagger:
                        algo = "ASkDAgger"
                    elif exp_folder == "exps":
                        algo = "ActiveDAgger"
                    elif exp_folder == "exps_thrifty":
                        algo = "ThriftyDAgger"
                    elif exp_folder == "exps_safe":
                        algo = "SafeDAgger"
                    else:
                        raise ValueError(f"Unknown exp_folder: {exp_folders}")
                    color = algos[algo]

                for iteration in range(10):
                    hydra.core.global_hydra.GlobalHydra.instance().clear()
                    os.chdir(os.environ["ASKDAGGER_ROOT"])
                    overrides_dict["iteration"] = iteration
                    overrides_dict["model_task"] = task
                    overrides_dict["eval_task"] = task
                    overrides = [f"{key}={value}" for key, value in overrides_dict.items()]
                    with initialize(config_path="../src/askdagger_cliport/cfg"):
                        vcfg = compose(
                            config_name="eval",
                            overrides=overrides,
                        )
                    OmegaConf.set_struct(vcfg, False)
                    train_results = utils.get_train_results(vcfg)
                    if train_results is not None:
                        r = train_results["r"]
                        demos = train_results["demos"]
                        episodes = train_results["episodes"]
                        demo_count = 0
                        max_i = 0
                        max_e = 0
                        n_relabeling = 0
                        success = []
                        n_annotation = 0
                        oracle_demos = []
                        stop = False
                        while demo_count < interactive_demos:
                            episode = episodes[max_e]
                            demo = False
                            for step in range(len(episode[3])):
                                if demos[max_i]:
                                    demo = True
                                    if r[max_i] == askdagger_cliport.KNOWN_FAILURE or (
                                        r[max_i] == askdagger_cliport.KNOWN_SUCCESS and not validation_demos
                                    ):
                                        n_annotation += 1
                                        oracle_demos.append(n_annotation)
                                        r_window = np.asarray(r[: max_i + 1])
                                        online_idx = np.asarray(r_window) >= -1
                                        novice_success = np.asarray(
                                            [
                                                episode[3][step] > 0
                                                for episode in episodes[: max_e + 1]
                                                for step in range(len(episode[3]))
                                            ]
                                        )
                                        success.append(np.sum(novice_success[-50:]) / len(novice_success[-50:]))
                                if max_i < len(r) - 1:
                                    if r[max_i + 1] == askdagger_cliport.UNKNOWN_RELABELING:
                                        demo = True
                                        n_relabeling += 1
                                        max_i += 1
                                max_i += 1
                            if demo:
                                demo_count += 1
                            max_e += 1
                            if demo_count < interactive_demos and max_e >= len(episodes):
                                stop = True
                                break
                        if stop:
                            print(f"Not enough demos collected, only {demo_count}/{interactive_demos}")
                            print(f"setting={setting}, task={task}, iteration={iteration}")
                        setting_results["success"].append(success)
                        setting_results["n_annotation"].append(oracle_demos)

                if len(setting_results["success"]) == 10:
                    label = algo if ax_idx == 0 else None


                    lens = [len(results) for results in setting_results["n_annotation"]]
                    lens.sort()
                    max_len = lens[4]
                    results = [result[:max_len] for result in setting_results["success"]]
                    mean_success, std_success = column_wise_mean_std(results)

                    ax[ax_idx].plot(
                        range(max_len),
                        mean_success,
                        color=color,
                        label=label,
                    )
                    ax[ax_idx].fill_between(
                        range(max_len),
                        mean_success - std_success,
                        mean_success + std_success,
                        alpha=0.2,
                        color=color,
                    )
                    if askdagger:
                        ax[ax_idx].set_ylim([0, max(mean_success + std_success) + 0.1])
                        ax[ax_idx].set_xlim(0, max_len)
                else:
                    print(f"setting={setting}, task={task} missing results")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
    plt.savefig("figures/training_scenario1.pdf")
