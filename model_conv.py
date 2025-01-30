import torch.nn as nn
class NgramWaveNet(torch.nn.Module):
    def __init__(self, vocab_size, n_embed, block_size,learning_rate, debug=False):
        super().__init__()
        # Add your code here!
        self.debug = debug
        #Wavenet has dilation factor of 1,2,4,8 for subsequent conv layers.
        self.dilation = [1, 2, 4, 8]
        #Create the embedding layer mapping from vocab_size to n_embed
        self.embedding = nn.Embedding(vocab_size, n_embed)
        #Create 3 one dimensional conv layers.
        #Also adjusting padding based on dilation.
        self.convs = nn.ModuleList([
            nn.Conv1d(n_embed, n_embed, kernel_size=2, dilation=l, padding=(1 * l)) for l in self.dilation])
        #Creating the norm layer
        self.layer_norm = nn.LayerNorm(n_embed)
        #self.batch_norm = torch.nn.BatchNorm1d(n_embed)- Tried batch norm but dimensions didnt work out
        self.dropout = nn.Dropout(0.5)
        #Creating the fully connected layer.
        self.fc = nn.Linear(n_embed, vocab_size)


    def forward(self, x):
        x = self.embedding(x)
        #Permuting to (N,C,L) format
        x = x.permute(0, 2, 1)
        for conv in self.convs:
            #Feeding the input to the conv layers
            x = conv(x)
            #Using Relu activation
            x=F.relu(x)
            x = self.dropout(x)

        x = self.layer_norm(x.permute(0, 2, 1))  # (N, L_out, C)
        x = x.mean(dim=1)  # resulting shape would be (N, C)

        output = self.fc(x)  #(N, vocab_size)
        return output