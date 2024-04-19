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

#url = "localhost:9000"
#options = [
#    ('grpc.max_message_length', 1024 * 1024 * 1024),
#    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
#    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
#]
#channel = grpc.insecure_channel(url, options=options)

train_file = "/Users/jdfld/Documents/Programmering/KEX/gridchunk1_1_1.h5ad"
val_file = "/Users/jdfld/Documents/Programmering/KEX/gridchunk1_1_2.h5ad"

 


def objective(config):
    training_data  = GeneDataset.GeneDataset(train_file,encoder=True)
    validation_data = GeneDataset.GeneDataset(val_file,encoder=training_data.get_encoder())

    net = model.BasicNeuralNetwork(training_data.genes,training_data.cells,training_data.get_encoder(),[512,512,512],config['lr'])
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
        with torch.no_grad():
            acc = net.predict_acc(*validation_data.get_batch())
            train.report({"mean_accuracy":acc})
# 512*60000 + 1024*2048 + 512*1024 + 512*256 + 256*128
ray.init()
param_space = {"lr":tune.loguniform(1e-4,1e-3)}#'layer_dims':tune.choice([[512, 512]])}

algo = OptunaSearch()

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        scheduler=ASHAScheduler(),
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
    ),
    run_config=train.RunConfig(
        stop={"training_iteration":10}
    ),
    param_space=param_space
)
#tuner.init()
result = tuner.fit()
print("best config is: ", result.get_best_result().config)
ray.shutdown()