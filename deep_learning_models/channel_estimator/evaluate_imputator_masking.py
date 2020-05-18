
import torch
import random
import torch.utils.data
from torch import optim
from torch import nn
from utils import DotDict, Mask, get_absolute, add_awgn
from train_imputator import argParser
from dataset_factory import dataset_factory
from models.data_imputator import DataImputator
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchsummary import summary
from tqdm import trange
import matplotlib2tikz
from models import DataImputator
import warnings
import argparse
from experimentlogger import load_experiment
from easydict import EasyDict as edict

def argparser():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--name', type=str, help='Name of experiment/model to load')
    parser.add_argument('--exp-folder', type=str, default='exps')
    args = parser.parse_args()
    return args

def plot_grid(input_tensor, predicted_tensor, true_tensor, title):
    input = input_tensor.cpu().detach().numpy()
    predicted = predicted_tensor.cpu().detach().numpy()
    true = true_tensor.cpu().detach().numpy()


    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,3,1)
    ax.set_title(title)
    pp = ax.imshow(input[0,0,:,:], aspect='auto')
    ax = plt.subplot(1,3,2)
    pp = ax.imshow(predicted[0,0,:,:], aspect='auto')
    ax = plt.subplot(1,3,3)
    pp = ax.imshow(true[0,0,:,:], aspect='auto')
    plt.show(block=False)

def plot_sequences(sequential, random, mc, std, title):
    sequential = sequential.cpu().detach().numpy()
    random = random.cpu().detach().numpy()
    mc = mc.cpu().detach().numpy()
    std = std.cpu().detach().numpy()


    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(1,4,1)
    ax.set_title(title[0])
    pp = ax.imshow(sequential[0,0,:,:], aspect='auto')
    ax = plt.subplot(1,4,2)
    ax.set_title(title[1])
    pp = ax.imshow(random[0,0,:,:], aspect='auto')
    ax = plt.subplot(1,4,3)
    ax.set_title(title[2])
    pp = ax.imshow(mc[0,0,:,:], aspect='auto')
    ax = plt.subplot(1,4,4)
    ax.set_title(title[3])
    pp = ax.imshow(std[0,0,:,:], aspect='auto')
    plt.show(block=False)



def run(args):

    
    exp_root_path = "exps/"

    exp = load_experiment(args.name, root_path = exp_root_path)
    name = args.name
    args = edict(exp.config)

    #args = DotDict(parameters)
    torch.manual_seed(args.seed)

    if args.cuda:
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    print(device)

    batch_size = args.batch_size
   


    # Load dataset
    train, test = dataset_factory(args.datadir, args.dataset, args.T, data_augment=False)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0)

    feature_dim = train.chunks_inputs.shape[1:4]
    args.channels = feature_dim[0]
    args.subcarriers = feature_dim[1]


    # Load Model
    imputator = torch.load(exp_root_path+"models/"+name+"_model.pt")
    imputator = imputator.to(device)
    imputator.eval()
    #summary(imputator, input_size=(feature_dim))
    criterion = nn.MSELoss()

    # Set SNR range
    SNRdB = [40]

    N_sc = args.subcarriers # Total number of available subcarriers
    Totsym_subframe = N_sc*14 # Total number of OFDM symbols per subframe
    max_oh = ((N_sc/2)/Totsym_subframe)*100
    
    print("Max OH if only 1 OFDM symbol per subframe is used: {}".format(max_oh))
    randommask = Mask(args.T, args.subcarriers, 'subcarriers',  mask_type='random', period=2)
    sequentialmask = Mask(args.T, args.subcarriers, 'subcarriers', mask_type='sequential', period=2)
    iterrange = 100
    # Set overhead range
    #oh_range = np.linspace(5, 5) # Percent
    oh_range = np.linspace(2, 6, 10)
    results_random = np.empty((len(testloader), len(SNRdB), len(oh_range)))
    results_uncertainty = np.empty((len(testloader), len(SNRdB), len(oh_range)))
    results_sequential = np.empty((len(testloader), len(SNRdB), len(oh_range)))
    results_srs = np.empty((len(testloader), len(SNRdB), len(oh_range)))
    for batch_idx, (g_tilde, h) in enumerate(testloader):
        if batch_idx == iterrange:
            break

        torch.cuda.manual_seed(batch_idx+int(args.seed))
        h_noise = h.to(device)
        h = h.to(device)
        g_tilde = g_tilde.to(device)
        #h_noise = h
        np.random.seed(batch_idx + int(args.seed))

        #loader = iter(testloader)
        print("Batch {}/{}".format(batch_idx, len(testloader)))
        #g_hat, h = loader.next() # Get next batch
        for snridx, SNR in enumerate(SNRdB):
            print("SNR: {}/{}".format(snridx, len(SNRdB)))

            #h_noise = add_awgn(h, SNR)


            for ohidx, oh in enumerate(oh_range):
                print("OH: {}/{}".format(ohidx, len(oh_range)))


                ###########################
                # Actual SRS sequence
                ###########################
                imputator = imputator.eval()
                predicted_srs = imputator(g_tilde)
                mseloss_srs = criterion(predicted_srs, h)
                results_srs[batch_idx, snridx, ohidx] = mseloss_srs.item()
                predicted_srs = predicted_srs.detach()
                g_tilde = g_tilde.detach()

                ############################
                # Sequential sequence
                ############################
                imputator = imputator.eval()
                masked_input_sequential = sequentialmask.mask_input(h_noise, oh)
                masked_input_sequential = masked_input_sequential.to(device)
                predicted_sequential = imputator(masked_input_sequential)
                mseloss_sequential = criterion(predicted_sequential, h)
                results_sequential[batch_idx, snridx, ohidx] = mseloss_sequential.item()
                masked_input_sequential = masked_input_sequential.detach()
                predicted_sequential = predicted_sequential.detach()

                #plot_grid(masked_input_sequential, predicted, h, 'Sequential')
                
                #############################
                # Randomly placed sequences
                #############################
                imputator = imputator.eval()
                masked_input_random = randommask.mask_input(h_noise, oh)
                masked_input_random = masked_input_random.to(device)
                predicted_random = imputator(masked_input_random)
                mseloss_random = criterion(predicted_random, h)
                results_random[batch_idx, snridx, ohidx] = mseloss_random.item()

                #plot_grid(masked_input_random, predicted, h, 'Random')

                # clear variables
                #h = h.detach()
                masked_input_random = masked_input_random.detach()
                predicted_random = predicted_random.detach()


                #############################
                # Uncertainty
                #############################

                # Place first sequence randomly
                masked_random_start = randommask.mask_input(h_noise, oh, num_sequences=1) # Place first randomly
                masked_random_start = masked_random_start.to(device)

                # Place the next sequences based on uncertainty
                subframes_remaining =  int(np.floor(h_noise.shape[3]/randommask.period)*randommask.period) # Round down
                subframes_to_sample = np.arange(9, subframes_remaining, step=10)
                imputator = imputator.eval()
                with torch.no_grad():
                    predicted_mc, std = imputator.MC(masked_random_start, num_samples = 100)

                predicted_mc = predicted_mc.to(device).detach()
                std = std.to(device).detach()



                for subframe in subframes_to_sample:

                    # Find subcarrier with maximum uncertanity
                    std_pow = torch.pow(std,2)  # Mean over std in real and imag
                    std_mean = torch.mean(std,dim=1)
                    max_sc = torch.argmax(std_mean[:,:,subframe], dim=1)
                    #max_std = torch.max(std[:,:,subframe], dim=1)
                    #sorted_peaks = torch.argsort(std[:,:,subframe], dim=1, descending=True)

                    # fig = plt.figure(figsize=(10, 8))
                    # plt.subplot(1,2,1)
                    # plt.plot(std_pow[0,0,subframe,:].cpu().detach().numpy())
                    # plt.subplot(1,2,2)
                    # plt.plot(std_pow[0,1,subframe,:].cpu().detach().numpy())
                    # plt.show()

                    # Find sequence_length on either side
                    # Loop for all batches
                    masked_random_start = masked_random_start.to(device)
                    for sc in range(h_noise.shape[0]):
                        sequence = randommask.find_span_peak(max_sc[sc].cpu())

                        #sequence = randommask.select_peaks(sorted_peaks[sc])

                        masked_random_start[sc,:,sequence,subframe] = h_noise[sc,:,sequence,subframe] # Set new sequence

                    # Predict using newly set sequence
                    masked_random_start = masked_random_start.to(device)
                    imputator = imputator.eval()
                    with torch.no_grad():
                        predicted_mc, std = imputator.MC(masked_random_start, num_samples = 100)

                    performance_after = criterion(predicted_mc.to(device), h.to(device))
                    std = std.to(device).detach()
                    predicted_mc = predicted_mc.to(device).detach()
                    masked_random_start = masked_random_start.to(device).detach()
                    torch.cuda.empty_cache()


                    #fig = plt.figure(figsize=(8,8))
                    #ax = plt.subplot(1,2,1)
                    #pp = ax.imshow(std[0,0,:,:].to(device).detach().numpy(), aspect='auto')
                    #ax = plt.subplot(1,2,2)
                    #pp = ax.imshow(masked_random_start[0,0,:,:].to(device).detach().numpy(), aspect='auto')
                    #plt.show()

                #plot_grid(masked_random_start, predicted, h, 'Uncertainty')
                results_uncertainty[batch_idx, snridx, ohidx] = performance_after.item()
                #plot_sequences(masked_input_sequential, masked_input_random, masked_random_start, std, ['Sequential', 'Random', 'Uncertainty', 'std'])
                #plot_sequences(predicted_sequential, predicted_random, predicted_mc, h, ['Sequential', 'Random', 'Uncertainty', 'True'])
                print("MSE random: {}".format(results_random[batch_idx, snridx, ohidx]))
                print("MSE sequential: {}".format(results_sequential[batch_idx, snridx, ohidx]))
                print("MSE uncertainty {}".format(results_uncertainty[batch_idx, snridx, ohidx]))
                print("MSE SRS {}".format(results_srs[batch_idx, snridx, ohidx]))
                #print("Total number of symbols with {}% OH : {}".format(oh, N_sym_pilots))
                #print("MSE with {}% OH;  SNR: {} = {}".format(oh, SNR, mseloss.item()))

    with plt.style.context('seaborn'):
        fig = plt.figure()
        plt.plot(oh_range, np.mean(results_uncertainty[:,0,:],axis=0),'-o', label='Uncertainty scheme')
        plt.plot(oh_range, np.mean(results_random[:, 0, :], axis=0), '-o', label='Random scheme')
        plt.plot(oh_range, np.mean(results_sequential[:, 0, :], axis=0), '-o', label='Sequential scheme')
        plt.plot(oh_range, np.mean(results_srs[:, 0, :], axis=0), '-o', label='Actual SRS')
        plt.xlabel('Overhead [%]')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(args.results_folder+'/oh_SNR{}.eps'.format(name))
        plt.savefig(args.results_folder+'/oh_SNR{}.png'.format(name))
        plt.show()

    np.save(args.results_folder+'/results_uncertainty_{}.npy'.format(name), results_uncertainty)
    np.save(args.results_folder+'/results_random_{}.npy'.format(name), results_random)
    np.save(args.results_folder+'/results_sequential_{}.npy'.format(name), results_sequential)


if __name__ == '__main__':
    p = argparser()
    print(p)
    run(p)
