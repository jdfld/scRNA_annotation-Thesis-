import sklearn as sk
import numpy as np
from torch import nn
import torch
from torch.utils import dataloader
from scipy import sparse

def align(mapping,data):
    # realigns data to fit mapping using matrix multiplication
    # uses more efficient sparse matrix mult.
    # requires data sorted according to gene order
    return sparse.csr_matrix.dot(data,mapping)

def map_genes(self,gene_expr):
    # map genes to the same format as training data as a matrix
    # the mapping is a sparse matrix 
    # that when multiplied with data transforms it into the same shape as the embedders training data
    # assumes data has been sorted along genes
    gene_count = len(gene_expr)
    i,j = 0,0
    mapping = np.zeros((gene_count,self.emb_count))
    while i < self.emb_count and j < gene_count:
        if self.emb_genes[i] == gene_expr[j]:
            mapping[i][j] = 1
            i += 1
            j += 1
        elif self.emb_genes[i] < gene_expr[j]:
            i += 1
        else:
            j += 1
    return sparse.csr_matrix(mapping)

def train_nn(data,model,optimizer,criterion,epochs):
    for epoch in range(epochs):
        for row in range(min(data.shape[0],4)):
            row_data = torch.tensor(data[row].toarray())
            recon = model(row_data)
            loss = criterion(recon,row_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')

class Encoder(nn.Module): # maybe will be used?
    
    def __init__(self,emb_genes):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.layer_dims = [self.emb_count,512,128,32]
        self.lr = 0.001
        encoder_layers = []
        for i in range(1,len(self.layer_dims)):
            if i > 1:
                encoder_layers.append(nn.Dropout(p=0.4))
            encoder_layers.append(nn.Linear(self.layer_dims[i-1],self.layer_dims[i]))
            encoder_layers.append(nn.BatchNorm2d(self.layer_dims[i]))
            encoder_layers.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(len(self.layer_dims)-2,-1,-1):
            decoder_layers.append(nn.BatchNorm2d(self.layer_dims[i+1]))
            if i > 0:
                decoder_layers.append(nn.Dropout(p=0.4))
            decoder_layers.append(nn.Linear(self.layer_dims[i+1],self.layer_dims[i]))
            encoder_layers.append(nn.BatchNorm2d(self.layer_dims[i]))
            decoder_layers.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        nn.init.kaiming_uniform_(self.encoder.params) # There is strange behaviour related to def init, unclear if patched
        nn.init.kaiming_uniform_(self.decoder.params)
   
            #outputs.append((epoch,batch,recon))
        
    def forward(self,x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def encode(self,x):
        return self.encoder(x)

    def embed(self,data,mapping=None):
        if mapping:
            data = align(mapping,data)
        assert data.shape[1] == self.emb_count

class BasicNeuralNetwork(nn.module): # basic neural network architecture with dropout and ReLu as activation
    def __init__(self,emb_genes,cell_types):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.cell_count = len(cell_types)
        self.cell_types = emb_genes
        self.layer_dims = [self.emb_count,512,512,512,self.cell_count]
        self.lr = 0.001
        model_layers = []
        for i in range(1,len(self.layer_dims)-1):
            if i > 1: # Do not want to drop from first layer due to sparse input
                model_layers.append(nn.Dropout(p=0.4))
            model_layers.append(nn.Linear(self.layer_dims[i-1],self.layer_dims[i]))
            model_layers.append(nn.BatchNorm2d(self.layer_dim[i-1]))
            model_layers.append(nn.LeakyReLU())
        model_layers.append(nn.Linear(self.layer_dims[-2],self.layer_dims[-1]))
        model_layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*model_layers)

        torch.nn.init.kaimeng_uniform_(self.model.parameters)

    def forward(self,x):
        return self.model(x)



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
        pred = self.model.predict(X)
        return sk.metrics.accuracy_score(y_true=y,y_pred=self.predict(X))

    def fit(self,X,y):
        y = self.encoder.fit_transform(y)
        if y is not np.array:
            y = y.toarray()
        self.model.fit(X,y)
        return sk.metrics.accuracy_score(y_true=y,y_pred=self.predict(X))
