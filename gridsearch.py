from ray import train,tune
from ray.tune.search.optuna import OptunaSearch
import ray
#from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
import model
import GeneDataset
import grpc

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

train_file = "C:/Users/random/Documents/KEX_data_stuff/gridchunk1_1_1.h5ad"
val_file = "C:/Users/random/Documents/KEX_data_stuff/gridchunk2_1_2.h5ad"

#monkey = "C:/Users/random/Documents/KEX_data_stuff/monkey_chunk.h5ad"
#frontotemporal = "C:/Users/random/Documents/KEX_data_stuff/frontotemporal_chunk.h5ad"
#striatum = "C:/Users/random/Documents/KEX_data_stuff/striatum_chunk.h5ad"
#anteriorcingulatecortex = "C:/Users/random/Documents/KEX_data_stuff/anteriorcingulatecortex_chunk.h5ad"




 


def objective(config):
    training_data  = GeneDataset.GeneDataset(train_file,encoder=True,batch_size=config["batch_size"])

    validation_data = GeneDataset.GeneDataset(val_file,encoder=training_data.get_encoder())

    #mon_data = GeneDataset.GeneDataset(monkey,encoder=True,batch_size=100)
    #frt_data = GeneDataset.GeneDataset(frontotemporal,encoder=True,batch_size=100)
    #str_data = GeneDataset.GeneDataset(striatum,encoder=True,batch_size=100)
    #ant_data = GeneDataset.GeneDataset(anteriorcingulatecortex,encoder=True,batch_size=100)


    layer_dims = []
    i = 0
    while "layer_dims_"+str(i) in config:
        i+=1
        layer_dims.append(config["layer_dims_"+str(i)])

    net = model.BasicNeuralNetwork(training_data.genes,training_data.cells,training_data.get_encoder(),layer_dims,config['lr'])
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        device = 'mps'
    net.to(device)
    
    


    # dot(Wq * in, Wk * in) * Wv*in 
    # Wq = Wk = t,m
    # Wv = g,m
     
    optimizer = torch.optim.Adam(net.parameters(),lr=net.lr)
    criterion = nn.NLLLoss()
    while True:
        net.train()
        loss = model.train_epoch(dataloader=training_data,model=net,label_type='sc',optimizer=optimizer,criterion=criterion)
        net.eval()

        # setup other networks
        net[:-1]




        with torch.no_grad():
            acc = net.predict_acc(*validation_data.get_batch())

            train.report({"mean_accuracy":acc})
# 512*60000 + 1024*2048 + 512*1024 + 512*256 + 256*128
ray.init()
param_space = {"lr":tune.loguniform(1e-4,1e-2),"batch_size":tune.qrandint(500,1000),"layer_dim_0":tune.qrandint(0,500),"layer_dim_1":tune.qrandint(0,500),"layer_dim_2":tune.qrandint(0,500),"layer_dim_3":tune.qrandint(0,500)}
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