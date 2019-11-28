# Algorithm components
import swiftwatcher.data_analysis as data
import swiftwatcher.algorithm as alg
import utils.video_io as vio


def main():
    """Execute each of the core functions of the swift-counting algorithm."""

    configs = vio.load_configs()

    for config in configs:
        if len(config["corners"]) == 2:
            events = alg.swift_counting_algorithm(config)

            if len(events) > 0:
                features = data.generate_feature_vectors(events)
                labels = data.generate_classifications(features)
                total = data.export_results(config, labels)
                print("[-]     Analysis complete. {} detected chimney swifts "
                      "in specified video.".format(total))
            else:
                print("[-]     Analysis complete. No detected chimney swifts "
                      "in specified video.")
        else:
            print("[!] Corners not selected for {}. Cannot process."
                  .format(config["name"]))


if __name__ == "__main__":
    main()
