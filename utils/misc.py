import os
import pandas as pd
import pdb
import yaml

def load_config(path : str) -> dict:
    """
    Loads and parses a YAML configuration file
    """
    with open(path, 'r', encoding='utf-8') as yamlfile:
        cfg = yaml.safe_load(yamlfile)

    return cfg
        

"""
This function formats the internal data properly if there are any issues.
- If the error type is "UnicodeDecodeError" there may be multiple underlying causes. Inspect manually using the trace.

"""
def load_and_save_data():
    base_path = 'data/internal'
    trials = os.listdir(base_path)
    for trial in trials:
        print(f"Trial: {trial}")
        path = f'{base_path}/{trial}/dm_suppdm_merged_sdtm.csv'
        try:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
            if len(df.columns) > 1:
                continue
                #df.to_csv(path, index=False, encoding='utf-8')
            else:
                pdb.set_trace()
                df = pd.read_csv(path, encoding='utf-8', delimiter='\t', low_memory=False)
                df.to_csv(path, index=False, encoding='utf-8')

        # handle exceptions
        except Exception as e:
            # handle delimiter error
            if type(e) == pd.errors.ParserError:
                df = pd.read_csv(path, delimiter='\t')
                df.to_csv(path, index=False)

            # handle encoding error
            elif type(e) == UnicodeDecodeError:
                ### note encoding='unicode_escape' works as well for 3013-05
                df = pd.read_csv(path, encoding='unicode_escape') #cp1252
                pdb.set_trace()
                df.to_csv(path, index=False)
                
            
            else:
                print("TRIAL: ", trial)
                pdb.set_trace()
                print(e)


if __name__ == '__main__':
    load_and_save_data()
 