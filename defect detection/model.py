import torch.nn as nn 
import torch

class Model(nn.Module):   
    def __init__(self, encoder , config ):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config= config
        self.sigmoid =  nn.Sigmoid()
     

        
    
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = self.sigmoid(outputs)
            return outputs