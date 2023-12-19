from vigipy.utils.misc import load_config
import pandas as pd
import os
import pdb
from vigipy.utils.data_prep import convert
from vigipy.PRR import prr
from vigipy.ROR import ror
from vigipy.GPS import gps
import matplotlib.pyplot as plt
import venn # pyvenn package

class StatisticalAnalyser:
     def __init__(self, config_path: str) -> None:
          self.CFG = load_config(config_path)
          self.blank_indicator = '[BLANK]'
          self.dataset_name = self.CFG['analysis']['dataset_name']
          self.df, self.labels = self.__LoadData()
          
     def __LoadData(self):
          df = pd.read_csv(self.CFG['data'][self.dataset_name]['feature_path'])
          labels = pd.read_csv(self.CFG['data'][self.dataset_name]['all_path'])['label']
          return df, labels
          
     def PreprocessData(self, feature_col):
          """
          Formats data to be compatible with vigipy and computes a contingency table
          """
          unique_terms = self.df[feature_col].unique().tolist()
          unique_terms = list(set([term.strip() for e in unique_terms for term in e.split(',')])) # account for multiple outcomes and AE's for a single entry
          contingency_data = {'AE': [], 'count': [], 'name': []}
          for term in unique_terms:
               if not len(term.strip()): # do not include blanks
                    continue
               # filter data and count occurences
               # if feature_col != 'dose':
               if feature_col in ['ade', 'outcome']: # account for multiple AE's
                    df_sub = self.df.loc[self.df[feature_col].str.strip().str.contains(term, regex=False)]
               else:
                    df_sub = self.df.loc[self.df[feature_col].str.strip().str.lower() == term.lower()]
               # df_sub = self.df[feature_col].apply(lambda x: fuzz.ratio(x, term) >= 90)
               # df_sub = df_sub[df_sub == True]
               # pdb.set_trace()
               labels_sub = self.labels.iloc[df_sub.index]
               N_positive = len(labels_sub[labels_sub == 1])
               N_negative = len(labels_sub[labels_sub == 0])

               # check if term is blank
               if len(term.strip()) == 0:
                    term = feature_col + f' {self.blank_indicator}'
               
               # log data
               contingency_data['name'].append(f'{feature_col}||' + term)
               contingency_data['count'].append(N_positive)
               contingency_data['AE'].append('Positive')
               contingency_data['name'].append(f'{feature_col}||' + term)
               contingency_data['AE'].append('Negative')
               contingency_data['count'].append(N_negative)

          # compute contingency table
          contingency_data = convert(pd.DataFrame.from_dict(contingency_data))
          return contingency_data

     def analyse(self, method: str):
          
          for i, feature_col in enumerate(self.CFG['analysis'][f'{self.dataset_name}_columns']):
               print(f'Performing analysis for {feature_col}')
               contingency_data = self.PreprocessData(feature_col)
               if method == 'PRR':
                    res = prr(contingency_data)
               
               elif method == 'MGPS':
                    res = gps(contingency_data)
               
               elif method == 'ROR':
                    res = ror(contingency_data)

               else:
                    print("The specified method is not supported")
                    raise NotImplementedError

               if i == 0:
                    result = res.all_signals
               else:
                    result = pd.concat([result, res.all_signals], ignore_index=True, axis=0)
          
          result = self.postprocess_results(result, method)
          return result
     
     def postprocess_results(self, result, method):
          # filter results to only keep significant effects
          threshold_col = self.CFG['analysis']['methods'][method]['threshold_col']
          threshold_val = self.CFG['analysis']['methods'][method]['threshold_value']
          result = result.loc[(result['Adverse Event'] == 'Positive') & (result[threshold_col] >= threshold_val) & (result['product margin'] >= 100)].reset_index(drop=True)
          L = [e.split('||') for e in result['Product']]
          result['Feature'] = [e[0] for e in L]
          result['value'] = [e[1] for e in L]
          result = result[['Feature', 'value', 'Adverse Event', threshold_col, 'p_value', 'Count', 'Expected Count', 'event margin', 'product margin']]
          result.to_csv(f'stat_analyses/results/{self.dataset_name}/{method}_analysis.csv', index=False)
          return result
     
     def __venn_diagram(self, results, ax):
          venn.venn(results, cmap="Set2", fontsize=9, legend_loc="lower right", ax=ax)
     
     def plot_results(self, ax):
          ## load results
          result_dir = f'stat_analyses/results/{self.dataset_name}'
          result_files = os.listdir(result_dir)
          results = {}
          if len(result_files) >= 3:
               for file in result_files:
                    method_name = file.split('_')[0]
                    df = pd.read_csv(f'{result_dir}/{file}')

                    results[method_name] = set(df['value'].values.tolist())
          else:
               print("You have yet to run all analyses! Please do so before attempting to plot results")
               return
          
          ## plot results
          fig = self.__venn_diagram(results, ax)

          return fig 


if __name__ == '__main__':
     SA_liverfailure = StatisticalAnalyser('configs/stats_config_liverfailure.yaml')
     SA_tramadol = StatisticalAnalyser('configs/stats_config_tramadol.yaml')

     # Run only the first time
     # SA_liverfailure.analyse('PRR')
     # SA_liverfailure.analyse('ROR')
     # SA_liverfailure.analyse('MGPS')
     # SA_tramadol.analyse('PRR')
     # SA_tramadol.analyse('ROR')
     # SA_tramadol.analyse('MGPS')


     # Create a figure with two subplots side by side
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))  # Adjust figsize as needed

     # Plot results in each subplot
     plt1 = SA_liverfailure.plot_results(ax1)
     plt2 = SA_tramadol.plot_results(ax2)

     # Optionally set titles for each subplot
     ax1.set_title("Liver Failure Analysis")
     ax2.set_title("Tramadol Analysis")


     plt1 = SA_liverfailure.plot_results(ax1)
     plt2 = SA_tramadol.plot_results(ax2)

     # Adjust layout and show the plot
     plt.tight_layout()
     plt.savefig('stat_analyses/results/legacy_venn_diagram.pdf', format='pdf', dpi=300)


     # # Two plots side 
     # fig, axs = plt.subplots(1, 2)
     
     # plt1 = SA.plot_results()
     # plt1.title('Liver failure')
     # plt1.ylabel('')
     # plt1.xlabel('')