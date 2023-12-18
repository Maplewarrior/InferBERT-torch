# Load your DataFrame
import pandas as pd

def make_root_latex_table(root_path, features):

    # df_part = pd.read_csv('experiments/reproduction/outputs/liverfailure/causality_output/root.csv')
    df_part = pd.read_csv(root_path)
    df_part = df_part.loc[df_part['value'].isin(features)]

    # Custom formatting function
    def format_float(value, decimals):
        return f"{value:.{decimals}f}"

    # Apply custom formatting to each column
    df_part['z score'] = df_part['z score'].apply(lambda x: format_float(x, 2))
    df_part['probability of do value'] = df_part['probability of do value'].apply(lambda x: format_float(x, 2))
    df_part['probability of not do value'] = df_part['probability of not do value'].apply(lambda x: format_float(x, 2))
    df_part['probability difference'] = df_part['probability difference'].apply(lambda x: format_float(x, 2))


    # Format 'p value' in scientific notation
    df_part['p value'] = df_part['p value'].apply(lambda x: '{:.2e}'.format(x))

    # change column names

        # Change column names
    new_column_names = {
        'Feature': 'Clinical category',
        'value': 'Clinical term',
        'z score': 'Z Score',
        'probability of do value': 'Avg. do probability',
        'probability of not do value': 'Avg. not do probability',
        'probability difference': 'Probability Difference',
        'p value': 'Adjusted p-value',
        'support': 'Support'
        # Add other columns here if needed
    }
    df_part.rename(columns=new_column_names, inplace=True)


    # Convert to LaTeX
    print(df_part.to_latex(index=False))

if __name__ == '__main__':
    root_path = 'experiments/reproduction/outputs/liverfailure_1/causality_output/root.csv'
    features = ['Acetaminophen', 'Death', '18-39', 'larger than 100 MG', 'Female']

    print("Analgesics-induced liver failure")
    make_root_latex_table(root_path, features)

    root_path = 'experiments/reproduction/outputs/tramadol_1/causality_output/root.csv'
    features = ['SUICIDE ATTEMPT', '40-64', 'Hydrocodone Bitartrate', 'Drug abuse', 'Male']

    print("Tramadol-related mortalities")
    make_root_latex_table(root_path, features)
