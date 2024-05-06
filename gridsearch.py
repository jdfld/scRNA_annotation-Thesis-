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

#path = "/Users/jdfld/Documents/Programmering/KEX/" # mac
path = "C:/Users/random/Documents/KEX_data_stuff/" # win

train_file = path+"subchunk1_2.h5ad"
val_file = path+"gridchunk1_1_2.h5ad"

monkey = path+"monkey_chunk.h5ad"
frontotemporal = path+"fronto_temporal_chunk.h5ad"
striatum = path+"striatum_chunk.h5ad"
anteriorcingulatecortex = path+"anteriorcingulatecortex_chunk.h5ad"




 


def objective(config):
    training_data  = GeneDataset.GeneDataset(train_file,encoder=True,batch_size=2000)
    validation_data = GeneDataset.GeneDataset(val_file,encoder=training_data.get_encoder(),shuffle=False)

    mon_data = GeneDataset.GeneDataset(monkey,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    frt_data = GeneDataset.GeneDataset(frontotemporal,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    str_data = GeneDataset.GeneDataset(striatum,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    ant_data = GeneDataset.GeneDataset(anteriorcingulatecortex,gene_location='feature_name',cell_type_location='cell_type',map_genes = training_data.genes,encoder=True,batch_size=100,shuffle=False)
    layer_dims = []
    i = 0
    
    param_sum = config["alpha"] + config["beta"] + config["gamma"] + config["delta"]
    in_width = round(8*3*(config["alpha"] / param_sum)+256)
    out_width = round(500*8*(config["beta"] / param_sum)+5048)
    width_decay = 0.087*8*(config["gamma"] / param_sum)+0.6
    depth = round(8*(config["delta"] / param_sum)+2)
    cost = len(training_data.genes)*in_width
    layer_dims.append(in_width)

    for i in range(depth):
        cost+=2*in_width*out_width*width_decay**i
        layer_dims.append(round(out_width*width_decay**i))
        layer_dims.append(in_width)
    layer_dims.append(config["final_layer"])
    print(layer_dims,cost)

    while "layer_dims_"+str(i) in config:
        layer_dims.append(config["layer_dims_"+str(i)])
        i+=1

    net = model.BasicNeuralNetwork(training_data.genes,training_data.cells,training_data.get_encoder(),layer_dims,config['lr'],double=True,norm='layer')
    
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
        loss = model.train_epoch(dataloader=training_data,model=net,optimizer=optimizer,criterion=criterion)
        mon_net.train()
        model.train_epoch(dataloader=mon_data,model=mon_net,optimizer=mon_opt,criterion=criterion,batches=1)
        mon_net.eval()
        frt_net.train()
        model.train_epoch(dataloader=frt_data,model=frt_net,optimizer=frt_opt,criterion=criterion,batches=1)
        frt_net.eval()
        str_net.train()
        model.train_epoch(dataloader=str_data,model=str_net,optimizer=str_opt,criterion=criterion,batches=1)
        str_net.eval()
        ant_net.train()
        model.train_epoch(dataloader=ant_data,model=ant_net,optimizer=ant_opt,criterion=criterion,batches=1)
        ant_net.eval()
        # setup other networks
        net.eval()


        with torch.no_grad():
            acc = 0
            val_acc = 0
            for f,l in validation_data:
                val_acc += net.predict_acc(f,l) / validation_data.no_batches
            mon_acc = mon_net.predict_acc(*mon_data.get_batch(range(1,mon_data.no_batches)))
            frt_acc = frt_net.predict_acc(*frt_data.get_batch(range(1,frt_data.no_batches)))
            str_acc = str_net.predict_acc(*str_data.get_batch(range(1,str_data.no_batches)))
            ant_acc = ant_net.predict_acc(*ant_data.get_batch(range(1,ant_data.no_batches)))
            tot_acc = (val_acc+mon_acc+frt_acc+str_acc+ant_acc)/5
            #tot_acc = val_acc
            train.report({"mean_accuracy":tot_acc,"cost":cost,"val_acc":val_acc,"str_acc":str_acc,"ant_acc":ant_acc,"frt_acc":frt_acc,"mon_acc":mon_acc})
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

#  alpha*(beta-1)+alpha*beta

# 

ray.init()
#param_space = {"lr":tune.loguniform(1e-4,1e-3),"layer_dims_0":tune.qrandint(128,512,16),"layer_dims_1":tune.qrandint(128,8192,256),"layer_dims_2":tune.qrandint(128,512,16),"layer_dims_3":tune.qrandint(128,512,16),"layer_dims_4":tune.qrandint(32,128,8)}
param_space = {"lr":tune.loguniform(1e-4,1e-3),"alpha":tune.uniform(1e-3,1),"beta":tune.uniform(1e-3,1),"gamma":tune.uniform(1e-3,0.5),"delta":tune.uniform(1e-3,1),"final_layer":tune.qrandint(32,256,8)}
#param_space = {"lr":tune.logunifrom}
algo = OptunaSearch()

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 2}),
    tune_config=tune.TuneConfig(
        scheduler=ASHAScheduler(),
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
        num_samples=40,
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