import argparse

from .electre_lib import read_csv, electre

def main():
    """ main processing entry point """

    args = parse_command_line()

    df_alternatives = read_csv(args.alternatives_csv)
    df_thresholds_weights = read_csv(args.thresholds_weights_csv)

    electre_handler = electre(df_alternatives, df_thresholds_weights)
    result_path = electre_handler.write_results()

    print('Results have been placed at {}'.format(result_path))

def parse_command_line():
    """ Returns inputs from command line interface

    Args:
        None
    Kwargs:
        None
    Returns:
        parse_args result
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("alternatives_csv", help="Path to CSV file containing the alternatives list with scores")
    parser.add_argument("thresholds_weights_csv", help="Path to CSV file containing thresholds and weights")
    return parser.parse_args()

if __name__ == '__main__':
    main()