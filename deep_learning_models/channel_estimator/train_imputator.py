import torch
import random
import torch.utils.data
import configargparse
from torch import optim
from torch import nn
from utils import DotDict
from dataset_factory import dataset_factory
from models import DataImputator
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchsummary import summary
from tqdm import trange
import time
from scipy import interpolate
from torch.utils.tensorboard import SummaryWriter
from experimentlogger import Experiment

def argParser():
    p = configargparse.ArgParser()
    
    p.add('-c', '--config', is_config_file=True, help='config file path', default='configs/6GHz_2-10_imputator.txt')
    p.add('--datadir', type=str, help='path to dataset', default='data')
    p.add('--dataset', type=str, help='dataset name', default='srs_1000nframes')
    p.add('--cuda', action="store_true", help='Enable/disable cuda', default=False)
    p.add('--T', type=int, help='Sequence length', default=150)
    p.add('--seed', default=30, help='Manual seed')
    p.add('--batch_size', type=int, help='Batch size', default=100)
    p.add('--evaluate', default=False, action="store_true")
    p.add('--hidden_dim', type=int, default=40, help='Hidden layer size')
    p.add('--dilation', type=int, default=4, help='Dilation')
    p.add('--max_epochs', type=int, default=40, help='Number of epochs')
    p.add('--learning_rate', type=float, default=0.01, help='Learning rate')
    p.add('--dropout', type=float, default=0.1, help='Dropout probability')
    p.add('--modelpath', type=str, default='')
    p.add('--draw_folder', type=str, default='runs')
    p.add('--results_folder', type=str, default='results')
    p.add('--imputator_path', type=str, default='')
    p.add('--future_samples', type=int, default=10)
    p.add('--weight_decay', type=float, default=0.001)
    p.add('--data_augment',  action="store_true", help='Enable/disable data augmentation', default=False)
    args = vars(p.parse_args())
    return args


def run(args):
    opt = DotDict(args)
    torch.manual_seed(opt.seed)
    
    if opt.cuda:
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    batch_size = opt.batch_size
    
    ##### DATA LOADERS

    train, test = dataset_factory(opt.datadir, opt.dataset, opt.T, data_augment=opt.data_augment)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0)

    feature_dim = train.chunks_inputs[:,:,:,:].shape

    opt.channels = feature_dim[1]
    x_dim = feature_dim[2]
    t_dim = feature_dim[3]
    z_dim = opt.hidden_dim


    ##### MODEL SETUP
    print('Setting up model')
    model = DataImputator(opt.channels, opt.hidden_dim, x_dim, t_dim, dropout=opt.dropout)
    model.to(device)
    summary(model, input_size=feature_dim[1:])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler_model = lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
    
    print('Testing forward pass:')
    (data, target) = iter(trainloader).next()

    input = data.to(device)
    target = target.to(device)

    print("Input shape: {}".format(input.shape))
    print("Target shape: {}".format(target.shape))

    out = model(input)
    print("Output shape: {}".format(out.shape))

    loss = criterion(out, target)
    print("Output loss: {}".format(loss))

    writer = SummaryWriter()
    #writer.add_graph(model)
    test_loss = []
    train_loss = []

    #### TRAINING LOOP
    fig = plt.figure(figsize=(20,5))
    for epoch in range(opt.max_epochs):
        model.train()
        with trange(len(trainloader)) as t:
            trainloss = 0
            model.train()
            for batch_idx, (g_tilde, h) in enumerate(trainloader):
                g_tilde = g_tilde.to(device) # Sparse SRS sequence
                h = h.to(device) # Full frequency response
                optimizer.zero_grad()

                h_tilde = model(g_tilde) # Data imputation, e.g. h estimation

                loss = criterion(h, h_tilde)
                
                loss.backward()
                trainloss += loss.item()
                optimizer.step()
                t.set_postfix(loss=loss.item(), epoch=epoch)
                t.update()
            
            model.eval()
            elapsed = 0
            testloss = 0
            for batch_idx, (g_tilde, h) in enumerate(testloader):
                
                g_tilde = g_tilde.to(device)
                h = h.to(device)
                current = time.time()
                h_tilde = model(g_tilde)
                elapsed += time.time() - current
                loss = criterion(h_tilde, h)
                testloss += loss.item()
            avg_trainloss = trainloss/len(trainloader)
            avg_testloss = testloss/len(testloader)
            train_loss.append(avg_trainloss)
            test_loss.append(avg_testloss)
            writer.add_scalar('train_loss', avg_trainloss, epoch)
            writer.add_scalar('test_loss', avg_testloss, epoch)
            t.set_postfix(loss=avg_trainloss, testloss=avg_testloss, epoch=epoch, avg_elapsed = elapsed/batch_idx, lr=optimizer.param_groups[0]['lr'])
            scheduler_model.step(testloss/len(testloader))
            # if epoch % 1 == 0:
            #     plt.clf()
                
            #     err = torch.abs(h[0,0,:,:]- h_tilde[0,0,:,:]).cpu().detach().numpy()
            #     ax = plt.subplot(151)
            #     ax.contour(g_tilde[0,0,:,:].cpu().detach().numpy(), levels=100)
            #     ax.set_xlabel('# SRS sequence')
            #     ax.set_ylabel('Subcarrier')
            #     ax.set_title("Input sequences Real $\hat{g}$")
            #     ax = plt.subplot(152)
            #     ax.contourf(h_tilde[0,0,:,:].cpu().detach().numpy())
            #     ax.set_xlabel('# SRS sequence')
            #     ax.set_ylabel('Subcarrier')
            #     ax.set_title("Predicted channel Real $\widetilde{h}$")
            #     ax = plt.subplot(153)
            #     ax.contourf(h[0,0,:,:].cpu().detach().numpy())
            #     ax.set_xlabel('# SRS sequence')
            #     ax.set_ylabel('Subcarrier')
            #     ax.set_title("True channel Real $h$")
                
      
            #     ax = plt.subplot(154)
            #     ax.plot(h_tilde[0,0,:,-1].cpu().detach().numpy(),'b-', label='Predicted')
            #     ax.plot(h[0,0,:,-1].cpu().detach().numpy(),'-o',label='True')
            #     ax.plot(g_tilde[0,0,:,-1].cpu().detach().numpy(),'o',label='Input')
            #     ax.set_xlabel('Subcarrier')
            #     ax.set_ylabel('Real(h[t])')
            #     ax.set_title("h[t] prediction and input")
            #     ax.legend()

            #     ax = plt.subplot(155)
            #     err_contour = ax.contourf(err)
            #     ax.set_title('Absolute error')
            #     ax.set_xlabel('# SRS sequence')
                #ax.set_ylabel('Subcarrier')
                #plt.colorbar(err_contour)

                

                #plt.tight_layout()

                #writer.add_figure('predicted',fig, epoch)
                
                #plt.savefig(opt.draw_folder+'\{}.png'.format(epoch))
        if optimizer.param_groups[0]['lr'] < 1e-7:
            break
                
    
    exp = Experiment('file', config=dict(opt), root_folder='exps/')
    results_dict = dict()
    results_dict['train_loss'] = train_loss
    results_dict['test_loss'] = test_loss
    exp.results = results_dict
    exp.save()
    
    torch.save(model, exp.root_folder+'/models/{}_model.pt'.format(exp.id))

if __name__ == '__main__':  
    args = argParser()
    print(args)
    run(args)
