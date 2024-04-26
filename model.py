import sklearn as sk
from sklearn.metrics import accuracy_score
import numpy as np
from torch import nn
import copy
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
    else:
        criterion = nn.NLLLoss()
    for epoch in range(start_epoch,epochs):
        loss = train_epoch(dataloader=dataloader,model=model,label_type=label_type,optimizer=optimizer,criterion=criterion)
        print('Epoch:', epoch,', Loss:', loss)
    model.eval()

def train_epoch(dataloader,model,label_type,optimizer,criterion,batches=-1):
    batch_count = 0
    for batch in dataloader:
        feature,label = batch
        pred = model(feature)
        if label_type == 'AutoEnc': 
            label = feature
        #elif label_type == 'sc':
        #    label = batch.obsm['label']
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_count == batches:
            break
        batch_count += 1
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
    # bad performance, may need more trials#
    def __init__(self,emb_genes,cell_types,layer_dims,lr):
        super().__init__()
        
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.cell_count = len(cell_types)
        self.cell_types = emb_genes
        self.layer_dims = [self.emb_count]+list(filter(lambda x:x!=0,layer_dims))+[self.cell_count]
        self.lr = lr
        self.avg_matrix = [None] * len(self.layer_dims)
        self.res_weight = torch.full((len(self.layer_dims),1),1)
        self.res_weight.requires_grad=True
        model_layers = []
        for i in range(1,len(self.layer_dims)):
            n,m = self.layer_dims[i-1],self.layer_dims[i]
            if n!=m:
                k = m//n
                t = m%n
                self.avg_matrix[i] = torch.sparse_csr_tensor(
                    [k*i for i in range(n-t)]+[(k+1)*i for i in range(n-t,n)],
                    list(range(m)),
                    [n/m]*m,size=(n,m))                
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
            x = layer(x)
            if self.avg_matrix != None:
                x += self.res_weight[i] * torch.mm(x,self.avg_matrix[i])
            else:
                x += self.res_weight[i] * x
        return self.model[-1](x)

    def predict(self,encoder,x):
        return encoder.inverse_transform(self.forward(x).detach().numpy().argmax())


class BasicNeuralNetwork(nn.Module): # basic neural network architecture with dropout and ReLu as activation
    def __init__(self,emb_genes,cell_types, encoder,layer_dims, lr,copy=False):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.cell_count = len(cell_types)
        self.cell_types = emb_genes
        self.lr = lr
        self.encoder = encoder
        if copy:
            self.layer_dims = None
            self.model = None
            return
        self.layer_dims = [self.emb_count]+list(filter(lambda x:x!=0,layer_dims))+[self.cell_count]
        model_layers = []
        for i in range(1,len(self.layer_dims)-1):
            if i > 1: # Do not want to drop from first layer due to sparse input
                model_layers.append(nn.Dropout(p=0.4))
            model_layers.append(nn.Linear(self.layer_dims[i-1],self.layer_dims[i]))
            model_layers.append(nn.LeakyReLU())
            model_layers.append(nn.BatchNorm1d(self.layer_dims[i]))
        model_layers.append(nn.Linear(self.layer_dims[-2],self.layer_dims[-1]))
        model_layers.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*model_layers)

        def init_kaiming(module):
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight,nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
        
        self.model.apply(init_kaiming)

        device = 'cpu'
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            device = 'mps'
        #self.to(device)

    def forward(self,x):
        return self.model(x)

    def predict_decode(self,x):
        y = self.predict(x)
        return self.encoder.inverse_transform(y)

    def predict(self,x):
        y = self.forward(x)
        y = y.cpu()
        return y.detach().argmax(dim=1).numpy()
    
    def copy_model(self,new_cells,new_encoder): # creates a shallow copy of the object itself
        new_model = BasicNeuralNetwork(self.emb_genes,new_cells,new_encoder,None,self.lr,copy=True)
        new_model.layer_dims = self.layer_dims[:-1]
        new_model.layer_dims.append(new_model.cell_count)
        new_model.model = self.model[:-2] # remove previous output layer
        new_model.model.append(nn.Linear(self.layer_dims[-2],new_model.cell_count)).append(nn.LogSoftmax(dim=1))
        new_model.to(next(self.parameters()).device)
        return new_model

    def predict_acc(self,X,y):
        y = y.cpu()
        return accuracy_score(y_true=y,y_pred=self.predict(X))


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
