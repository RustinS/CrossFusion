import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def plot_kaplan_meier_curve(all_risk_scores, all_censorships, all_event_times, dataset_name, plot_save_address):
    font_path = os.path.expanduser("~/Fonts/Times New Roman.ttf")
    fm.fontManager.addfont(font_path)

    threshold = np.median(all_risk_scores)
    high_risk_idx = all_risk_scores >= threshold
    low_risk_idx = all_risk_scores < threshold

    high_risk_event_times = all_event_times[high_risk_idx]
    high_risk_events = all_censorships[high_risk_idx] == 0

    low_risk_event_times = all_event_times[low_risk_idx]
    low_risk_events = all_censorships[low_risk_idx] == 0

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    kmf_high.fit(high_risk_event_times, event_observed=high_risk_events, label="High Risk")
    kmf_low.fit(low_risk_event_times, event_observed=low_risk_events, label="Low Risk")

    results = logrank_test(high_risk_event_times, low_risk_event_times, event_observed_A=high_risk_events, event_observed_B=low_risk_events)
    p_value = results.p_value

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5))

    kmf_high.plot_survival_function(
        ax=ax, ci_show=False, color="red", linewidth=1, show_censors=True, censor_styles={"marker": "+", "ms": 6}
    )

    kmf_low.plot_survival_function(
        ax=ax, ci_show=False, color="blue", linewidth=1, show_censors=True, censor_styles={"marker": "+", "ms": 6}
    )

    if dataset_name != "GBMLGG":
        plt.title(f"{dataset_name}\n(p-value = {p_value:.2e})", fontsize=25)
    else:
        plt.title(f"GM&LGG\n(p-value = {p_value:.2e})", fontsize=25)
    plt.xlabel("Timeline (months)", fontsize=15)
    plt.ylabel("Cumulative Proportion Surviving", fontsize=15)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.ylim(0, 1.05)
    plt.xlim(left=0)

    plt.legend(fontsize=13, loc="best")

    plt.savefig(plot_save_address, dpi=300, bbox_inches="tight")
    plt.close()
