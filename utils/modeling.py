import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast


class InferBERT(nn.Module):
    def __init__(self, CFG) -> None:
        super(InferBERT, self).__init__()
        self.CFG = CFG
        self.hidden_size = self.CFG['model']['hidden_size']
        self.output_dim = 1 if not self.CFG['model']['use_focal_loss'] else 2
        albert_config = AlbertConfig(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=self.CFG['model']['hidden_dropout_prob'],
                                     attention_probs_dropout_prob=self.CFG['model']['attention_dropout_prob'])
        albert_config = AlbertConfig(**{
                                        "_name_or_path": "albert-base-v2",
                                        "attention_probs_dropout_prob": self.CFG['model']['attention_dropout_prob'],
                                        "hidden_dropout_prob": self.CFG['model']['hidden_dropout_prob'],
                                        "hidden_size": self.CFG['model']['hidden_size'],
                                        "intermediate_size": self.CFG['model']['intermediate_size'],
                                        "num_attention_heads": self.CFG['model']['n_attention_heads'],
                                        "num_memory_blocks": self.CFG['model']['n_memory_blocks']})
        self.base = AlbertModel.from_pretrained(self.CFG['model']['model_version'], config=albert_config)
        self.dropout = nn.Dropout(p=self.CFG['model']['fc_dropout_prob'])
        self.output_layer = nn.Linear(self.hidden_size, self.output_dim)
        # self.alpha = torch.tensor([1.], requires_grad=True).cuda()
        # self.beta = torch.tensor([0.], requires_grad=True).cuda()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    @autocast() # run mixed precision
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Might be worthwhile to concatenate several last hidden states (https://github.com/huggingface/transformers/issues/1328) --> Experiment with this?
        """
        cls_reps = self.base(input_ids, attention_mask, token_type_ids)[1] # run ALBERT forward pass
        logits = self.output_layer(self.dropout(cls_reps)) # classification layer
        probs = self.sigmoid(logits) if not self.CFG['model']['use_focal_loss'] else self.softmax(logits) # get probabilities from logits
        #probs = self.sigmoid(self.alpha * logits - self.beta) # get probabilities from logits
        # probs = self.sigmoid(torch.tensor(logits[0].item(), 1-logits[0].item()))
        return {'logits' : logits,
                'probs' : probs}
    
    @autocast()
    def uncertainty_est_inference(self, input_ids, attention_mask, token_type_ids):
        """
        Run T forward calls with dropout
        """
        res = torch.empty(size=(self.CFG['causal_inference']['prediction']['uncertainty_est']['T'], input_ids.size(0))) # T x batch_size
        for i in range(self.CFG['causal_inference']['prediction']['uncertainty_est']['T']):
            cls_reps = self.base(input_ids, attention_mask, token_type_ids)[1] # run ALBERT forward pass
            logits = self.output_layer(self.dropout(cls_reps)) # classification layer
            probs = self.sigmoid(logits) # get probabilities from logits
            res[i] = probs.squeeze()
        
        means = torch.mean(res, dim=0)
        vars = torch.var(res, dim=0)
        return means, vars
    
def build_model(CFG):
    if CFG['model']['pretrained_ckpt'] is None:
        print(f'Model initialized from scratch')
        model = InferBERT(CFG)
    else:
        print('Model initialized from {path}'.format(path=CFG['model']['pretrained_ckpt'].split('/')[-1]))
        model = InferBERT(CFG)
        model.load_state_dict(torch.load(CFG['model']['pretrained_ckpt'], map_location=torch.device('cpu')))
    
    return model   

def build_optimizer(model, CFG):
    if CFG['training']['use_optimizer'] == 'Adam':
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=CFG['training']['optimization']['Adam']['lr'],
                                weight_decay=CFG['training']['optimization']['Adam']['weight_decay'],
                                betas=tuple(CFG['training']['optimization']['Adam']['betas']),
                                eps=CFG['training']['optimization']['Adam']['epsilon'])
    else:
        raise NotImplementedError()
    
    return optimizer

def build_lr_scheduler(CFG, optimizer):
    return get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps=CFG['training']['warmup_steps'],
                                           num_training_steps=CFG['training']['total_steps'])

def get_loss_func(CFG):
    if CFG['model']['n_classes'] == 1:
        loss_func = nn.BCEWithLogitsLoss(pos_weight=CFG['data']['pos_weight'])
    else:
        raise NotImplementedError
    return loss_func
