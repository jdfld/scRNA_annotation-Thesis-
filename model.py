import sklearn as sk
import numpy as np
from torch import nn
import torch

# improvements to NN:
# LR scheduling, currently the loss improvement is weirdly terrible
# it almost appears to be fully taught after just a single batch of 1000 samples, why? should require a lot more data to learn the couple of million parameters
# more data
# test on differently sampled data and data from other source.
# current model can achieve 98.6 % accuracy on unseen data
# when sampled from the same distribution (training, subchunk1_0) (validation, subchunk1_1)
# expects data as a a dataloader object
# implement plotting of fancy graphs for better understanding of results
def train(dataloader,model,label_type,epochs,start_epoch=0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=model.lr)
    if label_type == 'AutoEnc':
        criterion = nn.MSELoss()
    elif label_type == 'sc':
        criterion = nn.NLLLoss()
    for epoch in range(start_epoch,epochs):
        loss = train_epoch(dataloader=dataloader,model=model,label_type=label_type,optimizer=optimizer,criterion=criterion)
        print(f'Epoch:{epoch+1}, Loss:{loss:.4f}')
    model.eval()

def train_epoch(dataloader,model,label_type,optimizer,criterion):
    for batch in dataloader:
        feature = batch.X
        pred = model(feature)
        if label_type == 'AutoEnc': 
            label = feature
        elif label_type == 'sc':
            label = batch.obsm['label']
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


class AutoEncoder(nn.Module): # maybe will be used?
    
    def __init__(self,emb_genes):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.layer_dims = [self.emb_count,2048,64]
        self.lr = 0.005
        encoder_layers = []
        for i in range(1,len(self.layer_dims)):
            if i > 1:
                encoder_layers.append(nn.Dropout(p=0.4))
            encoder_layers.append(nn.Linear(self.layer_dims[i-1],self.layer_dims[i]))
            encoder_layers.append(nn.BatchNorm1d(self.layer_dims[i]))
            encoder_layers.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(len(self.layer_dims)-2,-1,-1):
            if i < len(self.layer_dims)-2 and i > 0:
                decoder_layers.append(nn.Dropout(p=0.4))
            decoder_layers.append(nn.Linear(self.layer_dims[i+1],self.layer_dims[i]))
            decoder_layers.append(nn.BatchNorm1d(self.layer_dims[i]))
            if i > 0:
                decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        def init_kaiming(module):
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight,nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
        
        self.encoder.apply(init_kaiming)
        self.decoder.apply(init_kaiming)

   
            #outputs.append((epoch,batch,recon))
        
    def forward(self,x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def encode(self,x):
        return self.encoder(x)

class ResidualNeuralNetwork(nn.Module):
    def __init__(self,emb_genes,cell_types,layer_dims,lr):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.cell_count = len(cell_types)
        self.cell_types = emb_genes
        self.layer_dims = [self.emb_count]+list(filter(lambda x:x!=0,layer_dims))+[self.cell_count]
        self.lr = lr
        self.avg_matrix = []
        model_layers = []
        for i in range(1,len(self.layer_dims)):
            n,m = self.layer_dims[i-1],self.layer_dims[i]
            if n == m:
                cur_matrix = torch.eye(n)
            else:
                cur_matrix = torch.zeros(n,m)
                k = m//n
                t = m%n
                for i in range(n-t):
                    cur_matrix[i][k*i:k*(i+1)] = n/m
                for i in range(n-t,n):
                    cur_matrix[i][(k+1)*i-n+t:(k+1)*(i+1)-n+t] = n/m
            self.avg_matrix.append(cur_matrix)
            single_layer = [nn.LeakyReLU(),nn.Dropout(p=0.5),nn.Linear(n,m),nn.BatchNorm1d(m)]
            model_layers.append(nn.Sequential(*single_layer))
        model_layers.append(nn.Sequential(nn.LogSoftmax(dim=1)))
        self.model = nn.ModuleList(model_layers)
        def init_kaiming(module):
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight,nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
        self.model.apply(init_kaiming)

    def forward(self,x):
        for i,layer in enumerate(self.model[:-1]):
            x = layer(x) + torch.matmul(x,self.avg_matrix[i])
        return self.model[-1](x)

    def predict(self,encoder,x):
        return encoder.inverse_transform(self.forward(x).detach().numpy().argmax(dim=1))


class BasicNeuralNetwork(nn.Module): # basic neural network architecture with dropout and ReLu as activation
    def __init__(self,emb_genes,cell_types, layer_dims, lr):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.cell_count = len(cell_types)
        self.cell_types = emb_genes
        self.layer_dims = [self.emb_count]+list(filter(lambda x:x!=0,layer_dims))+[self.cell_count]
        self.lr = lr
        model_layers = []
        for i in range(1,len(self.layer_dims)-1):
            if i > 1: # Do not want to drop from first layer due to sparse input
                model_layers.append(nn.Dropout(p=0.4))
            model_layers.append(nn.LeakyReLU())
            model_layers.append(nn.Linear(self.layer_dims[i-1],self.layer_dims[i]))
            model_layers.append(nn.BatchNorm1d(self.layer_dims[i]))
        model_layers.append(nn.LeakyReLU())
        model_layers.append(nn.Linear(self.layer_dims[-2],self.layer_dims[-1]))
        model_layers.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*model_layers)

        def init_kaiming(module):
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight,nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
        
        self.model.apply(init_kaiming)

    def forward(self,x):
        return self.model(x)

    def predict(self,encoder,x):
        return encoder.inverse_transform(self.forward(x).detach().numpy())


class SimpleModel:
    def __init__(self,model_type='rfc',params=None):
        if model_type == 'rfc':
            if params:
                self.model = sk.ensemble.RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'])
            else:
                self.model = sk.ensemble.RandomForestClassifier()
            self.encoder = sk.preprocessing.OneHotEncoder(handle_unknown='ignore')

    def predict(self,X):
        return self.model.predict(X)

    def predict_acc(self,X,y):
        y = self.encoder.transform(y)
        if y is not np.array:
            y = y.toarray()
        return sk.metrics.accuracy_score(y_true=y,y_pred=self.predict(X))

    def fit(self,X,y):
        y = self.encoder.fit_transform(y)
        if y is not np.array:
            y = y.toarray()
        self.model.fit(X,y)
        return sk.metrics.accuracy_score(y_true=y,y_pred=self.predict(X))
