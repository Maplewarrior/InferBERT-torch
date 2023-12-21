import pandas as pd

def venn_diagam_data(root_paths: list[str]):
    """Returns a dictionary of sets of enriched terms for each run
    
    Args:
        root_paths (list[str]): List of paths to root.csv files
        
    Returns:
        dict: Dictionary of sets of enriched terms for each run
    """
    N = len(root_paths)
    results = {}
    # find enriched terms for each run
    for i, file in enumerate(root_paths):
        df_root = pd.read_csv(file)
        results[i] = df_root['value'].values.tolist()
    
    results_sets = {f"R{key}": set(val) for key, val in results.items()}
    return results_sets

def robustness_evaluation_plot_data(root_paths: list[str]):
    """Returns data for plotting robustness evaluation
    
    Args:
        root_paths (list[str]): List of paths to root.csv files
    
    Returns:
        tuple: (x_axis, y_axis)
    """
    N = len(root_paths)
    results = {}
    # find enriched terms for each run
    for i, file in enumerate(root_paths):
        df_root = pd.read_csv(file)
        results[i] = df_root['value'].values.tolist()
    

    # lenght of the longest list
    max_len = max([len(val) for val in results.values()])
    # Plot axis 
    x_axis = list(range(1,max_len+5))
    y_axis = []

    for i in x_axis:
        
        # slice the results dict to only include the first i lists. 
        # The root should be sorted according to significance
        results_sliced = {key: val[:i] for key, val in results.items()}

        # Convert the first list to a set
        intersected_set = set(results_sliced[next(iter(results_sliced))])
        # Iterate over the remaining lists and find the intersection
        for key, val in results_sliced.items():
            intersected_set = intersected_set.intersection(val)
        # Convert the set to the list
        intersected_result = list(intersected_set)

        # Calculate the average number of common terms
        intersects_n = len(intersected_result)
        y_axis.append(intersects_n/i)

    return x_axis, y_axis


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
        print(terms)
        term_counts += len(set(terms).intersection(terms_across_runs))
    # return average
    return term_counts / N