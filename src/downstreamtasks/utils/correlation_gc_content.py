import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def correlation_gc_center_window_to_bio_tech_chart(
        window_index,
        window_gc_content,
        Xembeddings_biological,
        Xembeddings_technical,
        report_folder_path,
):
    center_focus_indices = [i for i, x in enumerate(window_index) if x == 1]
    gc_content_center_focus = np.array(window_gc_content)[center_focus_indices]
    bio_center_focus = Xembeddings_biological[center_focus_indices]
    tech_center_focus = Xembeddings_technical[center_focus_indices]

    bio_correlations = []
    tech_correlations = []

    for i in range(bio_center_focus.shape[1]):
        bio_corr, _ = pearsonr(bio_center_focus[:, i], gc_content_center_focus)
        bio_correlations.append(bio_corr)

    for i in range(tech_center_focus.shape[1]):
        tech_corr, _ = pearsonr(tech_center_focus[:, i], gc_content_center_focus)
        tech_correlations.append(tech_corr)

    # chart
    bio_correlations = np.array(bio_correlations)
    tech_correlations = np.array(tech_correlations)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for Biological Features
    #TODO: replace these with histograms
    axs[0].bar(range(bio_correlations.shape[0]), bio_correlations)
    axs[0].set_title('Biological Features')
    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Pearson Correlation with GC-content')

    # Plot for Technical Features
    axs[1].bar(range(tech_correlations.shape[0]), tech_correlations)
    axs[1].set_title('Technical Features')
    axs[1].set_xlabel('Feature Index')
    axs[1].set_ylabel('Pearson Correlation with GC-content')

    plt.tight_layout()
    file_name = "correlation_gc_center_window_to_bio_tech_chart.png"
    path_to_save = f"{report_folder_path}/{file_name}"
    plt.savefig(path_to_save)
    plt.close()
    return file_name, bio_correlations, tech_correlations


def correlation_gc_seq_to_bio_tech_agg_chart(
        X_bio_maxed,
        X_bio_avg,
        X_tech_maxed,
        X_tech_avg,
        gc_content_enhancer_agg,
        report_folder_path,
):
    # torch tensors to numpy arrays
    X_bio_maxed = X_bio_maxed.numpy()
    X_bio_avg = X_bio_avg.numpy()
    X_tech_maxed = X_tech_maxed.numpy()
    X_tech_avg = X_tech_avg.numpy()

    bio_maxed_corr, bio_avg_corr, tech_maxed_corr, tech_avg_corr = [], [], [], []

    for i in range(X_bio_maxed.shape[1]):
        bio_maxed_corr.append(pearsonr(X_bio_maxed[:, i], gc_content_enhancer_agg)[0])
        bio_avg_corr.append(pearsonr(X_bio_avg[:, i], gc_content_enhancer_agg)[0])

    for i in range(X_tech_maxed.shape[1]):
        tech_maxed_corr.append(pearsonr(X_tech_maxed[:, i], gc_content_enhancer_agg)[0])
        tech_avg_corr.append(pearsonr(X_tech_avg[:, i], gc_content_enhancer_agg)[0])

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot correlations
    axs[0, 0].bar(range(len(bio_maxed_corr)), bio_maxed_corr)
    axs[0, 0].set_title('Bio Maxed Features')

    axs[0, 1].bar(range(len(bio_avg_corr)), bio_avg_corr)
    axs[0, 1].set_title('Bio Avg Features')

    axs[1, 0].bar(range(len(tech_maxed_corr)), tech_maxed_corr)
    axs[1, 0].set_title('Tech Maxed Features')

    axs[1, 1].bar(range(len(tech_avg_corr)), tech_avg_corr)
    axs[1, 1].set_title('Tech Avg Features')

    # Labels and layout
    for ax in axs.flat:
        ax.set(xlabel='Feature Index', ylabel='Correlation with GC-content')

    plt.tight_layout()
    file_name = "correlation_gc_seq_to_bio_tech_agg_chart.png"
    path_to_save = f"{report_folder_path}/{file_name}"
    plt.savefig(path_to_save)
    plt.close()
    return file_name, bio_maxed_corr, bio_avg_corr, tech_maxed_corr, tech_avg_corr
