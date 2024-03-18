import sklearn as sk
import numpy as np
from torch import nn
import torch

# improvements to NN:
# LR scheduling, currently the loss improvement is weirdly terrible
# it almost appears to be fully taught after just a single batch of 1000 samples
# try smaller batch sizes
# fix issues with using GPUs
# would be nice to look into what would be necessary in order to actually run stuff on google cloud
# deeper/wider https://blog.research.google/2021/05/do-wide-and-deep-networks-learn-same.html
# more data
# perhaps more fancy techniques such as skip connections 
# 
# test on differently sampled data and data from other source.
# current model can achieve 98.6 % accuracy on unseen data
# when sampled from the same distribution (training, subchunk1_0) (validation, subchunk1_1)
# expects data as a a dataloader object
# implement plotting of fancy graphs for better understanding of results
def train_nn(dataloader,encoder,model,label_type,epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in dataloader:
            feature = batch.X
            pred = model(feature)
            if label_type == 'AutoEnc': 
                label = feature
            elif label_type == 'sc':
                label = torch.FloatTensor(encoder.transform(batch.obs['supercluster_term'].to_numpy()[:,None]).todense())
            loss = criterion(pred,label)
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

class BasicNeuralNetwork(nn.Module): # basic neural network architecture with dropout and ReLu as activation
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
            model_layers.append(nn.BatchNorm1d(self.layer_dims[i]))
            model_layers.append(nn.LeakyReLU())
        model_layers.append(nn.Linear(self.layer_dims[-2],self.layer_dims[-1]))
        model_layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*model_layers)

        def init_normal(module):
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight,nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
        
        self.apply(init_normal)

    def forward(self,x):
        #a_pred = encoder.inverse_transform(NeuralNetwork.forward(torch.FloatTensor(adata.X.todense())).detach().numpy())
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
