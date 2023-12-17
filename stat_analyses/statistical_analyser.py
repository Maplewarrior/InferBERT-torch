from utils.misc import load_config
import pandas as pd
import os
import pdb
from stat_analyses.vigipy.utils.data_prep import convert
from stat_analyses.vigipy.PRR import prr
from stat_analyses.vigipy.ROR import ror
from stat_analyses.vigipy.GPS import gps
import matplotlib.pyplot as plt
import venn # pyvenn package

class StatisticalAnalyser:
     def __init__(self, config_path: str, method: str) -> None:
          self.CFG = load_config(config_path)
          self.method = method
          self.blank_indicator = '[BLANK]'
          self.dataset_name = self.CFG['analysis']['dataset_name']
          self.df, self.labels = self.__LoadData()
          pdb.set_trace()
          
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

     def analyse(self):
          
          for i, feature_col in enumerate(self.CFG['analysis'][f'{self.dataset_name}_columns']):
               print(f'Performing analysis for {feature_col}')
               contingency_data = self.PreprocessData(feature_col)
               if self.method == 'PRR':
                    res = prr(contingency_data)
               
               elif self.method == 'MGPS':
                    res = gps(contingency_data)
               
               elif self.method == 'ROR':
                    res = ror(contingency_data)
               else:
                    print("The specified method is not supported")
                    raise NotImplementedError

               if i == 0:
                    result = res.all_signals
               else:
                    result = pd.concat([result, res.all_signals], ignore_index=True, axis=0)
          
          result = self.postprocess_results(result)
          return result
     
     def postprocess_results(self, result):
          # filter results to only keep significant effects
          threshold_col = self.CFG['analysis']['methods'][self.method]['threshold_col']
          threshold_val = self.CFG['analysis']['methods'][self.method]['threshold_value']
          result = result.loc[(result['Adverse Event'] == 'Positive') & (result[threshold_col] >= threshold_val) & (result['product margin'] >= 100)].reset_index(drop=True)
          L = [e.split('||') for e in result['Product']]
          result['Feature'] = [e[0] for e in L]
          result['value'] = [e[1] for e in L]
          result = result[['Feature', 'value', 'Adverse Event', threshold_col, 'p_value', 'Count', 'Expected Count', 'event margin', 'product margin']]
          result.to_csv(f'results/{self.dataset_name}/{self.method}_analysis.csv', index=False)
          return result
     
     def __venn_diagram(self, results):
          methods = list(results.keys())
          labels = venn.get_labels([results[method] for method in methods])
          if len(results) == 3:
               fig, ax = venn.venn3(labels, names=methods)
               fig.show()
          elif len(results) == 4:
               fig, ax = venn.venn4(labels, names=methods)
               ax.set_title(f'{self.dataset_name} - method comparsion')
               fig.show()
     
     def plot_results(self):
          ## load results
          result_dir = f'results/{self.dataset_name}'
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
          self.__venn_diagram(results)
          plt.show()