""" Attention functions """
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Attention(nn.Module):
    def __init__(self, query_size, key_size, out_size=1):
        super(Attention, self).__init__()
        self.query_projection = nn.Linear(query_size, out_size)
        self.key_projection = nn.Linear(key_size, out_size)
        self.v = nn.Parameter(torch.FloatTensor(out_size, 1))
        init.xavier_uniform_(self.v)

    def forward(self, query, key, value, attention_mask, attention_option="dot"):
        """Compute attention at each time step
            Args:
                query: shape `[(batch_size), bath_size, dim]`
                key: shape `[batch_size, time_step, dim]`
                value: shape `[batch_size, time_step, dim]`
                attention_mask: if not None, shape `[batch_size, 1, time_step]`
                attention_option: how to compute attention, either "additive" or "dot".
                    "additive": additive way (Bahdanau et al., ICLR'2015)
                    "dot": multiplicative way (Luong et al., EMNLP'2015)
            Returns: context vector and attention score
            """
        assert attention_option in ["additive", "dot"]

        if attention_option == "additive":
            score = self.additive_attention_score(query, key)
        else:
            score = self.dot_attention_score(query, key)

        if attention_mask is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            norm_score = self.prob_normalize(score, attention_mask)

        context = norm_score.matmul(value)
        context = context.squeeze(-2)
        attn_score = norm_score.squeeze(-2)

        return context, attn_score
    
    def prob_normalize(self, score, mask):
        score = score.masked_fill(mask == 0, -1e18)
        norm_score = F.softmax(score, dim=-1)
        return norm_score
    
    def additive_attention_score(self, query, key):
        """ Tips: 
            You may need to use self.query_projection, self.key_projection, self.v to complete your code
        """
        # You should complete your code below
        
        
        return

    def dot_attention_score(self, query, key):
        # You should complete your code below
        
        
        return
