import pandas as pd


def robustness_evaluation(root_paths: list[str]):
    N = len(root_paths)
    results = {}
    
    for i, file in enumerate(root_paths):
        df_root = pd.read_csv(file)
        results[i] = df_root['value'].values.tolist()

    terms_across_runs = set(results[0]).intersection(*list(results.values())[1:])
    return terms_across_runs / N