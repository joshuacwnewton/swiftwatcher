# Stdlib imports
import os
import csv

# Data science libraries
import numpy as np
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn
seaborn.set()


def save_test_config(args, params):
    """Save the parameters chosen for the given test of the algorithm. Some
    parameters include commas, so files are delimited with semicolons."""
    save_directory = args.default_dir+args.custom_dir
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    # Writing a summary of the parameters to a file
    with open(save_directory + "/parameters.csv", 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key in params.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(params[key])])


def save_test_results(args, df_estimation):
    """Save the bird count estimations from image processing to csv files.

    Count labels:
        0, frame_number
        1, total_birds
        2, total_matches
        3, appeared_from_chimney
        4, appeared_from_edge
        5, appeared_ambiguous (could be removed, future-proofing for now)
        6, disappeared_to_chimney
        7, disappeared_to_edge
        8, disappeared_ambiguous (could be removed, future-proofing for now)
        9, outlier_behavior

    Estimate array contains a 10th catch-all count, "segmentation_error"."""

    # Create save directory if it does not already exist
    save_directory = args.default_dir+args.custom_dir+"/results"
    if not os.path.isdir(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError:
            print("[!] Creation of the directory {0} failed."
                  .format(save_directory))

    print("[========================================================]")
    print("[*] Saving results of test to files.")

    if df_estimation is None:
        df_estimation = pd.read_csv(args.save_directory+"/full.csv",
                                    index_col="TMSTAMP")

    count_estimate = df_estimation.values

    # Load ground truth csv file into Pandas DataFrame
    df_groundtruth = pd.read_csv(args.default_dir+args.groundtruth,
                                 index_col="TMSTAMP")
    ground_truth = df_groundtruth.values

    # Comparing ground truth to estimated counts, frame by frame
    num_counts = count_estimate.shape[0]
    results_full = np.hstack((ground_truth[0:num_counts, 0:10],
                              count_estimate[:, 0:10])).astype(np.int)

    # Using columns 1:10 so that the "frame number" column is excluded
    error_full = count_estimate[:, 1:10] - ground_truth[0:num_counts, 1:10]

    # Calculating when counts were overestimated
    error_over = np.copy(error_full)
    error_over[error_over < 0] = 0

    # Calculating when counts were underestimated
    error_under = np.copy(error_full)
    error_under[error_under > 0] = 0

    # Summarizing the performance of the algorithm across all frames
    results_summary = {
        "count_true": np.sum(ground_truth[0:num_counts, 1:10], axis=0),
        "count_estimated": np.sum(count_estimate[:, 1:10], axis=0),
        "error_net": np.sum(error_full, axis=0),
        "error_overestimate": np.sum(error_over, axis=0),
        "error_underestimate": np.sum(error_under, axis=0),
        "error_total": np.sum(abs(error_full), axis=0),
    }

    # Writing the full results to a file
    np.savetxt(save_directory+"/full.csv", results_full,
               delimiter=';')

    # Writing a summary of the results to a file
    with open(save_directory+"/summary.csv", 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([" ", "SEGMNTS", "MATCHES",
                             "ENT_CHM", "ENT_FRM", "ENT_AMB",
                             "EXT_CHM", "EXT_FRM", "EXT_AMB", "OUTLIER"])
        for key in results_summary.keys():
            filewriter.writerow(["{}".format(key),
                                 "{}".format(results_summary[key][0]),
                                 "{}".format(results_summary[key][1]),
                                 "{}".format(results_summary[key][2]),
                                 "{}".format(results_summary[key][3]),
                                 "{}".format(results_summary[key][4]),
                                 "{}".format(results_summary[key][5]),
                                 "{}".format(results_summary[key][6]),
                                 "{}".format(results_summary[key][7]),
                                 "{}".format(results_summary[key][8])])

    print("[-] Results successfully saved to files.")


def plot_function_for_testing(df_estimate, df_groundtruth):
    """For a given pair of equal-length sequences, plot a comparison of the
    cumulative totals and save to an image. Used to compare running totals
    between bird count estimates and ground truth."""

    segments_es = df_estimate["SEGMNTS"].cumsum()
    segments_gt = df_groundtruth["SEGMNTS"].cumsum()

    exit_chimney_cumsum_es = df_estimate["EXT_CHM"].cumsum()
    exit_chimney_cumsum_gt = df_groundtruth["EXT_CHM"].cumsum()

    enter_chimney_cumsum_es = df_estimate["ENT_CHM"].cumsum()
    enter_chimney_cumsum_gt = df_groundtruth["ENT_CHM"].cumsum()

    fig1, ax1 = plt.subplots()
    segments_es.plot(ax=ax1)
    segments_gt.plot(ax=ax1)
    ax1.legend(["ESTIMATION", "GROUND TRUTH"])
    plt.savefig('segments_cumsum.png', bbox_inches='tight')

    fig2, ax2 = plt.subplots()
    exit_chimney_cumsum_es.plot(ax=ax2)
    exit_chimney_cumsum_gt.plot(ax=ax2)
    ax2.legend(["ESTIMATION", "GROUND TRUTH"])
    plt.savefig('exit_chimney.png', bbox_inches='tight')

    fig3, ax3 = plt.subplots()
    enter_chimney_cumsum_es.plot(ax=ax3)
    enter_chimney_cumsum_gt.plot(ax=ax3)
    ax3.legend(["ESTIMATION", "GROUND TRUTH"])
    plt.savefig('enter_chimney.png', bbox_inches='tight')
