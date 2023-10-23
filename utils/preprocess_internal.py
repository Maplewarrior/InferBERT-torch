import os
import numpy as np
import pandas as pd
import pdb
import datasets
from transformers import AlbertTokenizer#, AlbertModel
from misc import load_config
from itertools import product

"""
#################### This class handels all data preprocessing at once prior to training ####################
        Steps:      
                - Load and merge all data
                - Create text input from data
                - Tokenize text inputs
                - Save to csv
"""

def isNan(element):
    return element != element

class DataPreprocessor:

    def __init__(self, CFG) -> None:
        self.CFG = CFG
        self.data_path = CFG['data']['dir']

        ##### Internal keep columns
        self.ae_keep_columns = ['USUBJID', 'AEPTCD', 'AEDECOD', 'AEREL', 'AEOUT', 'AESPID'] # subject_id, ae_number, ae, causal_relation, outcome
        self.ec_keep_columns = ['USUBJID', 'ECTRT', 'ECDOSE', 'ECDOSU'] # subject_id, drug, dosage, dosage_unit
        self.dm_keep_columns = ['USUBJID', 'AGE', 'AGEU', 'SEX', 'COUNTRY', 'ARM'] # subject_id, arm
        self.trials = [e for e in os.listdir(self.data_path) if '3013' in e]
        #self.trials.remove('processed') # remove processed folder since it is not a trial

        self.df_internal = None
        self.df_external = None

        # load previously processed dataset if possible
    
        #if os.path.isfile(f'{self.data_path}/processed/internal_processed.csv'):
        #    self.df_internal = pd.read_csv(f'{self.data_path}/processed/internal_processed.csv', low_memory=False)

    def load_internal(self):
        # raise NotImplementedError
        # #### load all data ####
        self.df_internal = pd.DataFrame(columns=list(set(self.ae_keep_columns + self.dm_keep_columns)))
        self.df_drug = pd.DataFrame(columns=list(self.ec_keep_columns))
        n_ae = 0
        for i, trialname in enumerate(self.trials):
            df_ae = pd.read_csv(f'{self.data_path}/{trialname}/ae_suppae_merged_sdtm.csv', encoding='utf-8', low_memory=False)
            df_ae = df_ae.drop(columns = [e for e in df_ae.columns if e not in self.ae_keep_columns], axis=1)
            n_ae += len(df_ae)

            df_dm = pd.read_csv(f'{self.data_path}/{trialname}/dm_suppdm_merged_sdtm.csv', low_memory=False, encoding='utf-8')
            df_dm = df_dm.drop(columns=[e for e in df_dm.columns if e not in self.dm_keep_columns], axis=1)

            df_ae = df_ae.merge(df_dm, how='left', on='USUBJID')

            df_ec = pd.read_csv(f'{self.data_path}/{trialname}/ec_suppec_merged_sdtm.csv', low_memory=False, encoding='utf-8')
            df_ec = df_ec.drop(columns=[e for e in df_ec.columns if e not in self.ec_keep_columns], axis=1)

            self.df_drug = pd.concat([self.df_drug, df_ec], axis=0, ignore_index=True)
            self.df_internal = pd.concat([self.df_internal, df_ae], axis=0, ignore_index=True)
        
        self.join_ae_drug()
        
        # #### data cleaning, remove rows with missing values ###
        # self.remove_nans(columns=['ECTRT', 'AEDECOD', 'AEOUT'])
        
        # #self.df_internal.drop(labels=self.df_internal.loc[self.df_internal['AEOUT'] == 'UNKNOWN'].index, axis=0, inplace=True) # AEOUT should not be used as endpoint!
        # self.df_internal.reset_index(drop=True, inplace=True)
        
        # #### format data ####
        # self.df_internal.rename(columns={'ECTRT' : 'drug',
        #                                 'AEDECOD' : 'ae',
        #                                 'ECDOSE' : 'dose',
        #                                 'ECDOSU' : 'dose_unit',
        #                                 'AEREL' : 'relation',
        #                                 'AEOUT' : 'outcome',
        #                                 'AEPTCD' : 'drug_id',
        #                                 'SEX' : 'sex',
        #                                 'AGE' : 'age',
        #                                 'AGEU':'age_unit',
        #                                 'COUNTRY': 'country'},
        #                             inplace=True)
        # self.df_internal = self.df_internal[['USUBJID', 'ae', 'drug', 'drug_id', 'dose', 'dose_unit', 'sex', 'age', 'age_unit', 'country', 'relation', 'outcome', 'ARM']]
        # pdb.set_trace()
        # print("Succesfully loaded internal data!")
    
    def join_ae_drug(self):
        # remove AE entries which are dependent (Related AE's where several entries are made because of intensification/deintensification)
        self.df_internal = self.df_internal.drop_duplicates(subset=['USUBJID', 'AESPID'], ignore_index=True)

        subjects = list(set(self.df_drug['USUBJID'])) # get all unique subjects
        patient_drug = {s : list(set(self.df_drug['ECTRT'])) for s in subjects}

        drug_ae = {s: None for s in subjects}
        for subj, drugs in patient_drug.items(): # loop over all subjects
            AEs = list(set(self.df_internal.loc[self.df_internal['USUBJID'] == subj]['AEDECOD'])) # ALL AEs that the subject has experienced
            drug_ae[subj] = list(product(AEs, drugs))

        pdb.set_trace()
        print("HI")

    def remove_nans(self, columns : list):
        for column in columns:
            df_inv = self.df_internal.loc[isNan(self.df_internal[column]) == True]
            if len(df_inv) > 0:
                drop_indicies = list(df_inv.index)
                self.df_internal = self.df_internal.drop(labels=drop_indicies, axis=0)
                self.df_internal.reset_index(drop=True, inplace=True)
    
    def load_external(self):
        pass
    
    def generate_inputs(self, idx):
        patient_data = self.df_internal.iloc[idx]
        if patient_data.isna().any():
            patient_data = patient_data.fillna(value={col : 'N/A' for col in self.df_internal.columns}).copy()

        out = 'Patient {sex}, {age} {age_unit} from {country} takes {drug} {dose} {dose_unit} and experienced {ae} which lead to [MASK].'.format(sex=patient_data['sex'],
                                                                                                                                     age=patient_data['age'],
                                                                                                                                     age_unit=patient_data['age_unit'],
                                                                                                                                     country=patient_data['country'],
                                                                                                                                     drug=patient_data['drug'],
                                                                                                                                     dose=patient_data['dose'],
                                                                                                                                     dose_unit=patient_data['dose_unit'],
                                                                                                                                     ae=patient_data['ae'])
        # lowercase data
        if self.CFG['data']['is_lowercase'] == True:
            return out.lower()
        
        # [MASK] --> patient_data['outcome']
        return out
    
    def populate_with_inputs(self):
        inputs = []
        print("Generating text inputs...")
        for i in range(len(self.df_internal)):
            inputs.append(self.generate_inputs(i))
            if max(1, i) % 100000 == 0:
                print(f'Iter: {i}/{len(self.df_internal)}')

        self.df_internal['input'] = inputs

    def _tokenize_batch(self, batch):
        return self.tokenizer(batch['input'],
                       padding='max_length',
                       truncation=True,
                       max_length=self.max_seq_len,
                       return_tensors='pt')

    def explore_class_balances(self):
        mapping_dict = {'FATAL' : None,
                        'RECOVERED/RESOLVED' : None,
                        'RECOVERED/RECOVERED WITH SEQ':None,
                        'RECOVERED/RESOLVED WITH SEQUELAE' : None,
                        'RECOVERING/RESOLVING' : None,
                        'NOT RECOVERED/NOT RESOLVED' : None}
        return {e : len(self.df_internal.loc[self.df_internal['outcome'] == e]) for e in mapping_dict.keys()}
        

    def preprocess(self, save=False):

        if self.df_internal is None: # check if internal data has been loaded
            self.load_internal() # load internal data
            pdb.set_trace()
            self.populate_with_inputs() # generate transformer text inputs
        
        self.dataset_internal = datasets.Dataset.from_pandas(self.df_internal[['input', 'outcome']])
        mapping_dict = {'FATAL' : 0,
                        'RECOVERED/RESOLVED' : 1,
                        'RECOVERED/RECOVERED WITH SEQ':1,
                        'RECOVERED/RESOLVED WITH SEQUELAE' : 1,
                        'RECOVERING/RESOLVING' : 2,
                        'NOT RECOVERED/NOT RESOLVED' : 2}
        self.dataset_internal = self.dataset_internal.map(lambda example :  {'label' : [mapping_dict[e] for e in example['outcome']]}, batched=True)
        self.dataset_internal.to_csv('data/internal/processed/internal_untokenized.csv')


        """
        self.dataset_internal = self.dataset_internal.map(self._tokenize_batch, batched=True) # tokenize entire dataset
        # map clinical outcome to labels
        pdb.set_trace()
        # TODO : Revise with stakeholders regarding label mapping!
        
        self.dataset_internal = self.dataset_internal.map(lambda example :  {'label' : [mapping_dict[e] for e in example['outcome']]}, batched=True)
        self.dataset_internal.to_csv('data/internal/processed/internal_final.csv')
        self.df_final = pd.DataFrame.from_dict({'input_ids': self.dataset_internal['input_ids'],
                                                'attention_mask' : self.dataset_internal['attention_mask'],
                                                'token_type_ids' : self.dataset_internal['token_type_ids'],
                                                'label' : self.dataset_internal['label']})
        self.df_final.to_csv('data/internal/processed/internal_final.csv')
        """
    
    def stratified_split(self, train_split=0.8, save=False):
        assert train_split < 1, 'Train split has to be under 100%'
        assert os.path.exists('data/internal/processed/internal_untokenized.csv'), 'No processed dataset to split! Please run the preprocess method before proceeding.'
        df_all = pd.read_csv('data/internal/processed/internal_untokenized.csv', low_memory=False)

        val_split = test_split = (1 - train_split) / 2

        # shuffle dataset
        df_all = df_all.sample(frac=1, random_state=42)

        # Specify seed to always have the same split distribution between runs
        
        grouped_df = df_all.groupby(by=['label'])
        arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([t[1] for t in arr_list])
        test_ds = pd.concat([v[2] for v in arr_list])

        print("Data balance..")
        for split, df in {'train': train_ds, 'val': val_ds, 'test' : test_ds}.items():
            print("split: ", split)
            for label in range(3):
                print(f"Label: {label}")
                print(len(df.loc[df['label'] == label])/len(df))



        return train_ds, val_ds, test_ds


            
        


if __name__ == '__main__':
    CFG = load_config('configs/internal_config.yaml')
    DP = DataPreprocessor(CFG)
    DP.load_internal()
    pdb.set_trace()
    #train, val, test = DP.stratified_split()

    #DP.preprocess()
    #pdb.set_trace()
    #### 0.002 / 1000 tokens
    #### 200 tokens per query
    #(0.0015 / 5) * 10000
    #print((0.0015 / 5) * 10000)