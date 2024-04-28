from ray import train,tune
from ray.tune.search.optuna import OptunaSearch
import ray
#from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
import model
import GeneDataset
import data_proc_methods as dpm
import grpc
from copy import deepcopy
import numpy as np
# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# https://docs.ray.io/en/latest/tune/index.html
# params={'lr': 0.0005037405705207463, 'batch_size': 577, 'layer_dim_0': 383, 'layer_dim_1': 445, 'layer_dim_2': 171, 'layer_dim_3': 47}
#url = "localhost:9000"
#options = [
#    ('grpc.max_message_length', 1024 * 1024 * 1024),
#    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
#    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
#]
#channel = grpc.insecure_channel(url, options=options)

# path = "/Users/jdfld/Documents/Programmering/KEX/" # mac
path = "C:/Users/random/Documents/KEX_data_stuff/" # win

train_file = path+"gridchunk1_1_1.h5ad"
val_file = path+"gridchunk1_1_2.h5ad"

monkey = path+"monkey_chunk.h5ad"
frontotemporal = path+"fronto_temporal_chunk.h5ad"
striatum = path+"striatum_chunk.h5ad"
anteriorcingulatecortex = path+"anteriorcingulatecortex_chunk.h5ad"




 


def objective(config):
    training_data  = GeneDataset.GeneDataset(train_file,encoder=True,batch_size=1000)
    validation_data = GeneDataset.GeneDataset(val_file,encoder=training_data.get_encoder())

    mon_data = GeneDataset.GeneDataset(monkey,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    frt_data = GeneDataset.GeneDataset(frontotemporal,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    str_data = GeneDataset.GeneDataset(striatum,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    ant_data = GeneDataset.GeneDataset(anteriorcingulatecortex,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    layer_dims = []
    i = 0


    while "layer_dims_"+str(i) in config:
        layer_dims.append(config["layer_dims_"+str(i)])
        i+=1

    net = model.BasicNeuralNetwork(training_data.genes,training_data.cells,training_data.get_encoder(),layer_dims,config['lr'])
    
    mon_net = net.copy_model(mon_data.cells,mon_data.encoder)
    frt_net = net.copy_model(frt_data.cells,frt_data.encoder)
    str_net = net.copy_model(str_data.cells,str_data.encoder)
    ant_net = net.copy_model(str_data.cells,str_data.encoder)
    # dot(Wq * in, Wk * in) * Wv*in 
    # Wq = Wk = t,m
    # Wv = g,m
    rng = np.random.default_rng(seed=42)
    optimizer = torch.optim.Adam(net.parameters(),lr=net.lr)
    mon_opt = torch.optim.Adam(mon_net.parameters(),lr=mon_net.lr)
    frt_opt = torch.optim.Adam(frt_net.parameters(),lr=frt_net.lr)
    str_opt = torch.optim.Adam(str_net.parameters(),lr=str_net.lr)
    ant_opt = torch.optim.Adam(ant_net.parameters(),lr=ant_net.lr)
    criterion = nn.NLLLoss()
    while True:
        net.train()
        for parameters in net.model.parameters():
            parameters.requires_grad = True
        
        loss = model.train_epoch(dataloader=training_data,model=net,label_type='sc',optimizer=optimizer,criterion=criterion)
        mon_net.train()
        model.train_epoch(dataloader=mon_data,model=mon_net,label_type='sc',optimizer=mon_opt,criterion=criterion,batches=1)
        mon_net.eval()
        frt_net.train()
        model.train_epoch(dataloader=frt_data,model=frt_net,label_type='sc',optimizer=frt_opt,criterion=criterion,batches=1)
        frt_net.eval()
        str_net.train()
        model.train_epoch(dataloader=str_data,model=str_net,label_type='sc',optimizer=str_opt,criterion=criterion,batches=1)
        str_net.eval()
        ant_net.train()
        model.train_epoch(dataloader=ant_data,model=ant_net,label_type='sc',optimizer=ant_opt,criterion=criterion,batches=1)
        ant_net.eval()
        # setup other networks
        net.eval()


        with torch.no_grad():
            acc = net.predict_acc(*validation_data.get_batch()) / 5

            ind = rng.integers(1,mon_data.no_batches)
            acc += mon_net.predict_acc(*mon_data.get_batch(range(1,mon_data.no_batches))) / 5
            ind = rng.integers(1,frt_data.no_batches)
            acc += frt_net.predict_acc(*frt_data.get_batch(range(1,mon_data.no_batches))) / 5
            ind = rng.integers(1,str_data.no_batches)
            acc += str_net.predict_acc(*str_data.get_batch(range(1,mon_data.no_batches))) / 5
            ind = rng.integers(1,ant_data.no_batches)
            acc += ant_net.predict_acc(*ant_data.get_batch(range(1,mon_data.no_batches))) / 5

            train.report({"mean_accuracy":acc})
# 59480 * resolution + resolution * width
# 
# f.e 59480 * 512 + 512 * 29740 + 29740 * 512 + 512 * 14295
# init_width
# init_res
# init_depth 
# 
# 59480 * 512 + 512 * 20000 + 20000*256 + 256*
# 2 = width  
#  
#  width + resolution * depth = 2
ray.init()
param_space = {"lr":tune.loguniform(1e-4,1e-2),"layer_dims_0":tune.qrandint(256,768,24),"layer_dims_1":tune.qrandint(1024,8192,96),"layer_dims_2":tune.qrandint(256,768,24),"layer_dims_3":tune.qrandint(32,512,8)}
algo = OptunaSearch()

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 2}),
    tune_config=tune.TuneConfig(
        scheduler=ASHAScheduler(),
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
        num_samples=100,
    ),
    run_config=train.RunConfig(
        stop={"training_iteration":10},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=5,
        ),
    ),
    param_space=param_space
)
#tuner.init()
result = tuner.fit()
print("best config is: ", result.get_best_result().config)
ray.shutdown()