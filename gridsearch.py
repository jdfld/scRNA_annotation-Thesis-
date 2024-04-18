from ray import train,tune
from ray.tune.search.optuna import OptunaSearch
import ray
#from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
import model
import data_proc_methods as dpm
import grpc

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# https://docs.ray.io/en/latest/tune/index.html

url = "localhost:9000"
options = [
    ('grpc.max_message_length', 1024 * 1024 * 1024),
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
]
channel = grpc.insecure_channel(url, options=options)

train_file = "/Users/jdfld/Documents/Programmering/KEX/gridchunk1_1_1.h5ad"
val_file = "/Users/jdfld/Documents/Programmering/KEX/gridchunk1_1_2.h5ad"

 


def objective(config):
    training_data = dpm.load_data(train_file)
    encoder = dpm.get_encoder(training_data)
    train_loader = dpm.create_annloader(training_data,encoder=encoder,batch_size=10,use_cuda=torch.cuda.is_available())
    # validation data
    validation_data = dpm.load_data(val_file)
    val_loader = dpm.create_annloader(validation_data,encoder=encoder,batch_size=10,use_cuda=torch.cuda.is_available())

    genes = training_data.var['Gene'] 
    cells = training_data.obs['supercluster_term'].cat.categories

    net = model.BasicNeuralNetwork(genes,cells,[1],config['lr'])
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)


    # dot(Wq * in, Wk * in) * Wv*in 
    # Wq = Wk = t,m
    # Wv = g,m

    optimizer = torch.optim.Adam(net.parameters(),lr=net.lr)
    criterion = nn.NLLLoss()
    while True:
        net.train()
        loss = model.train_epoch(dataloader=train_loader,model=net,label_type='sc',optimizer=optimizer,criterion=criterion)
        net.eval()
        with torch.no_grad():
            acc = net.predict(val_loader)
            train.report({"mean_accuracy":acc})
# 512*60000 + 1024*2048 + 512*1024 + 512*256 + 256*128
ray.init()
param_space = {"lr":tune.choice([1])}#'layer_dims':tune.choice([[512, 512]])}

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
        stop={"training_iteration":5}
    ),
    param_space=param_space
)
#tuner.init()
result = tuner.fit()
print("best config is: ", result.get_best_result().config)
ray.shutdown()