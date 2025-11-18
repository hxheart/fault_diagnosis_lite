import sys
from typing import List, Set
sys.path.append("../")

import torch
import math
from nutils import *

def decoder(num_values):
    class Decoder(torch.nn.Module):
        def __init__(self, hidden_dim, sliced):
            super().__init__()
            self.num_values = num_values
            self.sliced=sliced

            self.decoder_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.num_values),
            )
        
        def forward(self, x):
            x = x
            x = self.decoder_net(x)
            x = x.log_softmax(dim=-1)

            if self.sliced: return x.sum(axis=1)
            else: return x

        def decode(self, x, return_entropies=False):
            x = self.forward(x)
            res = x.argmax(axis=-1)

            if return_entropies:
                return res, torch.distributions.Categorical(logits=x).entropy()

            return res

        def loss(self, x, target):
            x = self.forward(x)
            mask = (target != -1).view(-1)
            x_masked = x.view(-1,self.num_values)[mask,:]
            target_masked = target.view(-1)[mask]
            return torch.nn.functional.nll_loss(x_masked, target_masked)
        
        def accuracy(self, x, target):
            x = self.forward(x).argmax(axis=-1)
            mask = (target != -1).view(-1)
            x_masked = x.view(-1)[mask]
            target_masked = target.view(-1)[mask]
            return (x_masked == target_masked).sum().float() / mask.sum()
        
        def f1(self, x, target):
            x = self.forward(x).argmax(axis=-1)
            mask = (target != -1).view(-1)
            x_masked = x.view(-1)[mask]
            target_masked = target.view(-1)[mask]
            return f1_loss(target_masked, x_masked)

    return Decoder

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    """

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def boolean_encoding():
    return onehot_encoding(2)

def binary_encoding(num_values):
    class BinaryEncoding(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()

            bitwidth = torch.ceil(torch.log2(torch.tensor(num_values).float())).long().item()
            self.embedding = torch.nn.Linear(bitwidth, hidden_dim, bias=False)
            
            mask = 2 ** torch.arange(bitwidth)
            self.register_buffer('mask', mask)
        
        def forward(self, x):
            mask = self.mask.clone().detach().requires_grad_(False)
            x = x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
            x = self.embedding(x)
            return x
    
    return BinaryEncoding, decoder(num_values)

def onehot_encoding(num_values):
    class OneHotEncoding(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(num_values + 1, hidden_dim)
        
        def forward(self, x):
            return self.embedding.forward(x.to(torch.long))
    
    return OneHotEncoding, decoder(num_values)

class MaskedFeatureEmbedding(torch.nn.Module):
    def __init__(self, embeddings: List[torch.nn.Module], hidden_dim, excluded_feature_indices: Set[int]):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        for i,e in enumerate(self.embeddings):
            self.add_module("embedding_" + str(i), e)
        self.num_embeddings = len(self.embeddings)
        self.excluded_feature_indices = list(excluded_feature_indices)
        self.mask_embedding = torch.nn.Parameter(torch.randn([len(self.embeddings), hidden_dim]))

    def forward(self, x, mask, reliable_masking):
        assert x.size(2) == self.num_embeddings, f"cannot embed input with shape {x.shape} using {self.num_embeddings} embeddings"
        
        x_emb = torch.zeros([x.size(0), x.size(1), self.hidden_dim], device=x.device)
        
        modules = dict(self.named_modules())

        for i in range(self.num_embeddings):
            embedding = modules['embedding_' + str(i)]
            if i in self.excluded_feature_indices: continue
           
            mask_emb = self.mask_embedding[i].unsqueeze(0).unsqueeze(0)
            feature_emb = embedding.forward(torch.maximum(torch.tensor(0), x[:,:,i] + 1))
            random_emb = feature_emb[torch.randint(0, feature_emb.size(0), device=x.device, size=[x.size(0)])]

            if not reliable_masking:
                masked_emb = categorical([feature_emb, random_emb, mask_emb], [0.1, 0.1, 0.8])
            else:
                masked_emb = mask_emb
            non_masked_emb = feature_emb

            feature_is_applicable = (x[:,:,i] != -1)
            feature_is_queried = (x[:,:,i] == -2)
            is_masked = mask[:,:,i].logical_or(feature_is_queried).unsqueeze(-1)

            x_emb = x_emb + feature_is_applicable.unsqueeze(-1) * (is_masked * masked_emb + is_masked.logical_not() * non_masked_emb)
        return x_emb

def mask_like(x, p=0.5):
    return ((torch.rand_like(x, dtype=torch.float) - (1-p)) >= 0)

class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=4*30000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        pos = self.pe[0, :x.size(0)].clone().detach().requires_grad_(False).unsqueeze(1)
        x = x + pos
        return self.dropout(x)

class NodeFeatureEmbedding(torch.nn.Module):
    def __init__(self, hidden_dim, features, excluded_feature_indices=set()):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(hidden_dim, 0.0)

        self.embedding = MaskedFeatureEmbedding([
            f.encoder(hidden_dim) for f in features
        ], self.hidden_dim, excluded_feature_indices)
    
    def forward(self, x, mask, reliable_masking, positional_encoding=True):
        x_emb = self.embedding.forward(x, mask, reliable_masking)
        
        if positional_encoding:
            x_emb = self.positional_encoding.forward(x_emb)

        return x_emb

class NodeFeatureDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, features, excluded_feature_indices=set(), sliced=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.excluded_feature_indices = excluded_feature_indices
        self.features_by_name = {}
        self.decoders = {}
        
        for i,f in enumerate(features):
            self.features_by_name[f.name] = f
            self.decoders[f.name] = f.decoder(hidden_dim, sliced=sliced)
            self.add_module(f"decoder_{f.name}_{i}" , self.decoders[f.name])

    def forward(self, x, feature_name):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]
        
        return decoder.forward(x)

    def decode(self, x, feature_name, return_entropies=False):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]
        
        return decoder.decode(x, return_entropies)

    def loss(self, x, target, feature_name):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]
        
        assert feature.idx not in self.excluded_feature_indices, f"cannot apply loss for excluded feature {feature.name}"
        loss = decoder.loss(x, target[:,:,feature.idx])
        if torch.isnan(loss): 
            print("nan loss for", feature_name)
            return torch.tensor(0)
        return loss
        
    def accuracy(self, x, target, feature_name):
        assert feature_name in self.features_by_name.keys(), f"unknown node feature {feature_name}, available {str(self.features_by_name.keys())}"
        feature: Feature = self.features_by_name[feature_name]
        decoder: Decoder = self.decoders[feature_name]

        assert feature.idx not in self.excluded_feature_indices, f"cannot evaluate accuracy for excluded feature {feature.name}"
        return decoder.accuracy(x, target[:,:,feature.idx])

    
