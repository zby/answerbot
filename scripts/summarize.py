import pandas as pd
import argparse
import numpy as np

def process_experiment_results(filename, column_names):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(filename)

    # Default aggfunc dictionary
    aggfunc_dict = {'correct': 'sum', 'error': 'sum', 'steps': 'mean'}

    # If column_names is not empty - make a pivot table
    if column_names:
        # All provided column_names should not be in ['correct', 'error', 'steps']
        if all(column_name not in ['correct', 'error', 'steps'] for column_name in column_names):
            pivot = df.pivot_table(index=column_names,
                                   values=['correct', 'error', 'steps'],
                                   aggfunc=aggfunc_dict)
            print(pivot)
        else:
            raise ValueError("Column names cannot include 'correct', 'error' or 'steps'.")
    # If no column names provided - print simple summary
    else:
        summary = df.agg(aggfunc_dict)
        print(summary)

def main():
    parser = argparse.ArgumentParser(description='Process experiment results.')
    parser.add_argument('--file', type=str, required=True, help='The CSV file path.')
    parser.add_argument('--columns', type=str, nargs='*', default=[], help='Column name(s) to pivot on.')
    args = parser.parse_args()

    process_experiment_results(args.file, args.columns)

if __name__ == '__main__':
    main()