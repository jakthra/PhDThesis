import torch
import numpy as np
import warnings
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_absolute(array):
    """
    Returns the absolute value of array. [N, M, K, L] where M are the real and imaginary part in that order
    """
    real = array[:,0,:,:]
    imag = array[:,1,:,:]
    return np.abs(real + 1j*imag)

def add_awgn(h, SNR):
    """
        Add Gaussian noise to the spectrum h
    """
    # Get complex numpy array
    h_complex = get_absolute(h.cpu().numpy())
    # Get power of h
    power = np.sum(np.conj(h_complex)*h_complex, 2)/h_complex.shape[2] # Average over time
    power = np.sum(power,1)/power.shape[1] # Average over frequency
    
    power_db = 10*np.log10(power) # Compute power in db
    noise_snr = power_db - SNR # compute noise SNR
    noise_snr_w = 10 ** (noise_snr/10) # compute noise SNR in W
    noise_std = torch.FloatTensor(np.sqrt(noise_snr_w)) # compute sigma of noise
    noise = torch.mul(noise_std, torch.randn(h.shape[1], h.shape[2], h.shape[3], h.shape[0])) # generate matrix of noise
    noise = noise.permute(3,0,1,2) # Fix dimensions to that of h
    h_noise = h.cpu() + noise # Add noise
    return h_noise

class Mask():

    def __init__(self, T, subcarriers, oh_type, mask_type='random', M = 2, period = 10):
        """
        shape : [subcarrier, T]
        oh_type : ['symbols', 'subcarriers']
        """
        self.oh_type = oh_type
        self.subcarriers = subcarriers
        self.T = T
        self.M = 2
        self.period = 10
        self.mask_type = mask_type

        self.Totsym_subframe = self.subcarriers * 14 # 14 OFDM symbols per subframe
        

    def mask_input(self, x, oh, num_sequences=None):
        """
        num_sequences : [None] - default none
        """
        masked = torch.zeros(x.shape).cpu()
        x = x.cpu()
        
        if self.oh_type == 'symbols':
            N_sym_pilots = self.Totsym_subframe * (oh/100) # Compute number of symbols in a subframe
            self.sequence_span = N_sym_pilots*self.M # Number of symbols spanned for the sequence
        
        elif self.oh_type == 'subcarriers':
            total_sub = (self.subcarriers * self.T)/self.M # total number of subcarriers given the period 
            total_sub_available = total_sub*(oh/100) # Number of subcarriers available given an OH percentage
            total_sub_period = np.floor(total_sub_available/self.period) # Number of subcarriers every 10 ms for a given OH percentag
            self.sequence_span = total_sub_period*self.M

        if self.sequence_span > self.subcarriers:
            warnings.warn('Requires more subcarriers ({}) than available, capped at {}'.format(self.sequence_span, self.subcarriers))
            self.sequence_span = self.subcarriers

        if num_sequences == None:
            num_sequences = int(np.floor(self.T/self.period))

        if self.mask_type == 'random':
            masked = self._random_mask(x, masked, num_sequences)
        else:
            masked = self._sequential_mask(x, masked, num_sequences)

        return masked

    def _sequential_mask(self, x, masked, num_sequences):
        seq_start = np.empty((num_sequences, 1))
        seq_end = np.empty((num_sequences, 1))
        # First one is placed randomly
        if self.sequence_span >= self.subcarriers:
            seq_start[0] = 0
            seq_end[0] = self.subcarriers-1
        else:
            seq_start[0] = np.random.randint(0, self.subcarriers-self.sequence_span-1, size=(1, 1))
            seq_end[0] = seq_start[0]+self.sequence_span-1

        # Next one is offset by the sequence span
        for seq in range(1,num_sequences):
            seq_start[seq] = seq_end[seq-1]
            seq_end[seq] = seq_start[seq] + self.sequence_span-1
            if seq_end[seq] > self.subcarriers:
                seq_start[seq] = 0
                seq_end[seq] = seq_start[seq] + self.sequence_span-1

        masked = self._mask_input(x, masked, num_sequences, seq_start, seq_end)

        return masked


    def _mask_input(self, x, masked, num_sequences, seq_start, seq_end):
        seq_count = 0
        for subframe in range(self.T):
            if subframe % self.period == 0:
                if seq_count >= num_sequences:
                    break
                

                
                seq = np.arange(seq_start[seq_count], seq_end[seq_count], step=self.M, dtype='int')
                masked[:, :, seq, subframe] = x[:, :, seq, subframe]

                seq_count += 1
        return masked

    def _random_mask(self, x, masked, num_sequences):

        if self.sequence_span >= self.subcarriers:
            seq_start = np.zeros((num_sequences, 1))
            seq_end = seq_start+self.subcarriers-1
        else:
            seq_start = np.random.randint(0, self.subcarriers-self.sequence_span, size=(num_sequences, 1))
            seq_end = seq_start+self.sequence_span

        masked = self._mask_input(x, masked, num_sequences, seq_start, seq_end)

        return masked

    def find_span_peak(self, max_sc):
        if max_sc-self.sequence_span/2 < 0:
            left_side_span = int(max_sc/self.M)
            right_side_span = self.sequence_span-left_side_span
        elif max_sc + self.sequence_span/2 > self.subcarriers:
            # it spans beyond the range of subcarriers, thus shift to either side
            
            diff_right = int((self.subcarriers-max_sc)/self.M)
            right_side_span = self.subcarriers-1
            left_side_span = max_sc-(self.sequence_span/2+diff_right)
        else:
            left_side_span = int(max_sc - self.sequence_span/2)
            right_side_span = int(max_sc + self.sequence_span/2)

        new_seq_span = np.arange(left_side_span, right_side_span, step=self.M)
        return new_seq_span

    def select_peaks(self, peaks):
        if self.sequence_span > len(peaks):
            selected_peaks = peaks
        else:
            selected_peaks = peaks[int(self.sequence_span)-1]
        return selected_peaks