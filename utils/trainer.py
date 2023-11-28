import json
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.misc import load_config
from utils.modeling import build_model, build_optimizer, build_lr_scheduler
from utils.dataset import ADRDataset
import pdb

class Trainer:
    def __init__(self, cfg_path):
        ################ Initialization ################
        self.CFG = load_config(cfg_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device depending on availability
        self.__init_logs()
        self.__init_data()
        loss_weight = torch.ones([self.CFG['model']['n_classes']]) * len(self.df_train.loc[self.df_train['label'] == 0]) / len(self.df_train.loc[self.df_train['label'] == 1])
        print(f'Loss weight: {loss_weight}')
        self.__init_modelling(loss_weight=None)
        print("Training modules initialized!")

    def __init_modelling(self, loss_weight=None):
        ################ Load model and dependencies ################
        self.model = build_model(self.CFG).to(self.device)
        # self.initial_sd = self.model.state_dict().copy()
        self.optimizer = build_optimizer(self.model, self.CFG)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        self.scaler = GradScaler()
        self.scheduler = build_lr_scheduler(self.CFG, self.optimizer)

    def __init_data(self):
        ################ Load data and create dataloaders ################ 
        ### train subset is based on: 1 = [Oxycodone, Acetaminophen] and 0 = [, Ibuprofen]
        # pos_idxs_train = [302, 338, 506, 511, 591, 646]
        # neg_idxs_train = [1838, 2047, 2084, 2879, 3155, 3167]
        # additional_indicies = list(range(1000, 1150))

        self.df_train = pd.read_csv(self.CFG['data']['train_path'])#.iloc[pos_idxs_train + neg_idxs_train + additional_indicies].reset_index(drop=True)
        self.df_val = pd.read_csv(self.CFG['data']['val_path'])#[150:180].reset_index(drop=True)
        self.df_train.astype({'sentence' : str,
                              'label': int})
        self.df_val.astype({'sentence' : str,
                              'label': int})

        self.dataloader_train = DataLoader(ADRDataset(df=self.df_train, CFG=self.CFG), 
                                            batch_size=self.CFG['training']['mini_batch_size'], num_workers=self.CFG['training']['num_workers'], shuffle=self.CFG['training']['shuffle'])
        
        self.dataloader_val = DataLoader(ADRDataset(df=self.df_val, CFG=self.CFG), 
                                            batch_size=self.CFG['training']['validation']['batch_size'], num_workers=self.CFG['training']['num_workers'], shuffle=self.CFG['training']['validation']['shuffle'])
    
    def __init_logs(self):
        ################ Create directories ################
        if not os.path.exists(self.CFG['training']['out_dir']):
            os.makedirs(self.CFG['training']['out_dir'])
            os.makedirs(self.CFG['training']['out_dir'] +'/logs')

        if self.CFG['training']['start_epoch'] == 0: # training from scratch
            self.train_results = {'loss' : [],
                                  'acc' : []}
            self.val_results = {'loss' : [],
                                'acc' : []}
        else: # continued training
            self.train_results = json.load(self.CFG['training']['train_results_path'])
            self.val_results = json.load(self.CFG['training']['val_results_path'])

    def train(self):
        n_epochs = int(np.ceil(self.CFG['training']['total_steps'] / len(self.dataloader_train)*self.CFG['training']['accum_iters']))
        step_count = 0 # initialize step counter
        best_loss = np.inf
        print(f'Starting training!\nTrain size: {len(self.df_train)}')
        for i in range(self.CFG['training']['start_epoch'], n_epochs):
            print(f'\nEpoch: {i+1}/{n_epochs}')
            ######### train
            train_loss, train_acc, step_count = self.train_1_epoch(step_count)
            self.train_results['loss'].append(train_loss)
            self.train_results['acc'].append(train_acc)
            save_logs(self.CFG['training']['train_results_path'], self.train_results)
            ######### validate
            val_loss, val_acc = self.evaluate()
            self.val_results['loss'].append(val_loss)
            self.val_results['acc'].append(val_acc)
            save_logs(self.CFG['training']['val_results_path'], self.val_results)
            
            ########## save model
            if np.mean(val_loss) < best_loss:
                best_loss = np.mean(val_loss)
                # if os.path.exists(self.CFG['training']['model_path']):
                    # os.remove(self.CFG['training']['model_path']) # remove current model (needed for DDL)
                torch.save(self.model.state_dict(), self.CFG['training']['model_path']) # save new model
                print('Model saved to {path}'.format(path=self.CFG['training']['model_path']))

            ######### stop training if total steps is reached
            if step_count % self.CFG['training']['total_steps'] == 0:
                break
    
    def train_1_epoch_vanilla(self):
        self.model.train()
        losses = []
        acc = 0
        accum_iters = self.CFG['training']['accum_iters']
        
        for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader_train):
            input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)

            out = self.model(input_ids, attn_masks, token_type_ids)
            loss = self.criterion(out['logits'].squeeze(1), labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            acc += compute_accuracy(out['probs'], labels)
            
            # print performance and duration
            if (i+1) % self.CFG['training']['print_freq'] == 0:
                print(f'Iter: {i+1}/{len(self.dataloader_train)}\nLoss: {np.mean(losses)}\nAcc: {acc / ((i+1) * labels.size(0))}')
            
            
        accuracy = acc/len(self.dataloader_train.dataset)
        print(f'\nFinal training performance:\n### Loss: {np.mean(losses)}\n### Accuracy: {accuracy}')
        return losses, accuracy

    
    def train_1_epoch(self, step_count):
        self.model.train()
        losses = []
        acc = 0
        accum_iters = self.CFG['training']['accum_iters']
        for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader_train):
            input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
            with autocast():
                out = self.model(input_ids, attn_masks, token_type_ids)
                loss = self.criterion(out['logits'], labels.unsqueeze(1))
            loss = loss / accum_iters # normalize loss since it is averaged (reduction='mean' by default)
            losses.append(loss.item())
            
            acc += compute_accuracy(out['probs'], labels)
            self.scaler.scale(loss).backward()

            if (i+1) % accum_iters == 0:
                # perform optimization step
                ### https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples ---> Link to docs about mixed precision training
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step_count += 1
            
            # print performance and duration
            if (i+1) % self.CFG['training']['print_freq'] == 0:
                print(f'Iter: {i+1}/{len(self.dataloader_train)}\nLoss: {np.mean(losses)}\nAcc: {acc / ((i+1) * labels.size(0))}')
            
            if max(1, step_count) % self.CFG['training']['total_steps'] == 0:
                accuracy = acc / ((i+1) * labels.size(0)) # divide by iter * batch_size
                return losses, accuracy, step_count
            
        accuracy = acc/len(self.dataloader_train.dataset)
        print(f'\nFinal training performance:\n### Loss: {np.mean(losses)}\n### Accuracy: {accuracy}')
        return losses, accuracy, step_count

    def evaluate(self):
        import time
        self.model.eval()
        losses = []
        acc = 0
        start = time.time()
        with torch.no_grad():
            for i, (input_ids, attn_masks, token_type_ids, labels) in enumerate(self.dataloader_val):
                input_ids, attn_masks, token_type_ids, labels = input_ids.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                out = self.model(input_ids, attn_masks, token_type_ids)
                loss = self.criterion(out['logits'], labels.unsqueeze(1)).item()
                losses.append(loss)
                acc += compute_accuracy(out['probs'], labels)

        accuracy = acc/len(self.dataloader_val.dataset)
        print(f'\nValidation performance:\n### Loss: {np.mean(losses)}\n### Accuracy: {accuracy}')
        end = time.time()
        print(f'Elapsed: {end-start}')
        return losses, accuracy

def save_logs(save_path, result):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

def compute_accuracy(preds, targets):
    preds = (preds >= 0.5).to(torch.int8) # convert to bool and [0, 1] thereafter
    acc = 0
    # count n.o. correct
    for i in range(preds.size(0)):
        if preds[i].item() == int(targets[i]):
            acc+=1
    # print(f'Predictions: {[e.item() for e in preds]}')
    return acc