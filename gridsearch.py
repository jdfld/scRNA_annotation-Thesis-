from ray import train,tune
from ray.tune.search.optuna import OptunaSearch
#from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
import model
import data_proc_methods as dpm

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# https://docs.ray.io/en/latest/tune/index.html

train_file = "gridchunk1_1_1.h5ad"
val_file = "gridchunk1_1_2.h5ad"

 
training_data = dpm.load_data(train_file)
encoder = dpm.get_encoder(training_data)
train_loader = dpm.create_annloader(training_data,encoder=encoder,batch_size=4000,use_cuda=torch.cuda.is_available())
# validation data
validation_data = dpm.load_data(val_file)
val_loader = dpm.create_annloader(validation_data,encoder=encoder,batch_size=4000,use_cuda=torch.cuda.is_available())

genes = training_data.var['Gene'] 
cells = training_data.obs['supercluster_term'].cat.categories

def objective(config):
    resnet = model.ResidualNeuralNetwork(genes,cells,[512,512],config['lr'])
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    resnet.to(device)


    # dot(Wq * in, Wk * in) * Wv*in 
    # Wq = Wk = t,m
    # Wv = g,m

    optimizer = torch.optim.Adam(model.parameters(),lr=model.lr)
    criterion = nn.NLLLoss()
    while True:
        loss = model.train_epoch(dataloader=train_loader,model=resnet,label_type='sc',optimizer=optimizer,criterion=criterion)
        with torch.nograd():
            acc = resnet.predict(val_loader)
            train.report({"mean_accuracy":acc})
# 512*60000 + 1024*2048 + 512*1024 + 512*256 + 256*128
param_space = {"lr":tune.loguniform(1e-4, 1e-3)}#'layer_dims':tune.choice([[512, 512]])}

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

result = tuner.fit()
print("best config is: ", result.get_best_result().config)