import pandas as pd


def robustness_evaluation(root_paths: list[str]):
    N = len(root_paths)
    results = {}
    # find enriched terms for each run
    for i, file in enumerate(root_paths):
        df_root = pd.read_csv(file)
        results[i] = df_root['value'].values.tolist()
    # find common enriched terms across all runs
    terms_across_runs = set(results[0]).intersection(*list(results.values())[1:])
    # find occurences of common enriched terms for each run
    term_counts = 0
    for run in results.keys():
        terms = results[run]
        term_counts += len(set(terms).intersection(terms_across_runs))
    # return average
    return term_counts / N