from typing import Any, Dict
import torch
from torch import nn

from determined import pytorch

import logging
logger = logging.getLogger(__name__)

##
## FlattenConsecutive from week3 of the course
##
class FlattenConsecutive(torch.nn.Module):
  
  def __init__(self, n):
    super().__init__()
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []

class NgramWaveNet(torch.nn.Module):
    def __init__(self, vocab_size, hparams: Dict):
        super().__init__()
        self.debug = False
        self.vocab_size = vocab_size
        n_embed = self.n_embed = hparams['n_embed']  # Embedding size
        self.block_size = hparams['block_size']      # Number of input channels (8)
        n_hidden = hparams['n_hidden']               # Number of hidden units
        
        self.emb = torch.nn.Embedding(self.vocab_size, self.n_embed)
        self.emb.weight.data *= hparams['embed_weight_data']
        
        self.model = nn.Sequential(
            self.emb,
            FlattenConsecutive(2), nn.Linear(n_embed * 2, n_hidden, bias=False), nn.LayerNorm(n_hidden), nn.Tanh(),
            nn.Dropout1d(hparams["dropout1"]),
            FlattenConsecutive(2), nn.Linear(n_hidden*2, n_hidden, bias=False), nn.LayerNorm(n_hidden), nn.Tanh(),
            nn.Dropout1d(hparams["dropout2"]),
            FlattenConsecutive(2), nn.Linear(n_hidden*2, n_hidden, bias=False), nn.LayerNorm(n_hidden), nn.Tanh(),
            nn.Dropout1d(hparams["dropout3"]),
            nn.Linear(n_hidden, self.vocab_size)
        )
        with torch.no_grad():
          self.model[-1].weight *= 0.1 # last layer make less confident
        logger.info(f"Model has {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, *args: pytorch.TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        
        output = self.model(x)
        return output
    
def build_model(vocab_size, hparams: Dict) -> nn.Module:
    return NgramWaveNet(vocab_size, hparams)