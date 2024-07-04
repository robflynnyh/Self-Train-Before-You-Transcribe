from omegaconf import OmegaConf
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Callable
import torch.optim as optim

from lcasr.utils.augmentation import SpecAugment # lcasr https://github.com/robflynnyh/long-context-asr
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.optim import madgrad
import random
from einops import rearrange
from lcasr.decoding import ctc_beam_search as beam_search
try:
        from lming.utils import general
except:
        general = None
        
import lcasr
from functools import partial
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage
from torch.nn import functional as F
from apex.normalization import FusedLayerNorm
from lcasr.components.batchrenorm import BatchRenorm1d
import time

def load_beamsearch(
        path:str,
        alpha:float=0.45,
        beta:float=1.53,
        prune_less_than_val:float=3.17,
        top_am_threshold:float=-6,
    ):
    assert general != None, 'install https://github.com/robflynnyh/language_modelling to use beam_search'
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint['model'] = general.convert_from_ddp(checkpoint['model'])
    model_config = checkpoint['config']
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = general.load_model(config=model_config, vocab_size=tokenizer.vocab_size())
    model.load_state_dict(checkpoint['model'], strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()
    language_model = beam_search.LanguageModel(
        model = model,
        bos_id = tokenizer.bos_id(),
        device = device,
    )
    BeamSearch = partial(
        beam_search.BeamSearch,
        language_model = language_model,
        tokenizer = tokenizer,
        #beam_width = 3,
        blank_id = tokenizer.vocab_size(),
        alpha = alpha,
        beta = beta,
        debug = False,
        prune_less_than_val = prune_less_than_val,
        top_am_threshold = top_am_threshold,
        max_cache_length = 128
    )
    return BeamSearch

def replace_with_frame(spec):
    for i, s in enumerate(spec):
        random_index = random.randint(0, spec.shape[-1])
        # set all frames to random frame
        spec[i] = spec[i, :, :] * 0 + spec[i, :, random_index, None]
    return spec

def frame_shuffle(spec, time_dimension=False, freq_dimension=False): # shuffles all frames in a spectrogram
    if time_dimension: spec = spec[:, :, torch.randperm(spec.shape[-1])]
    if freq_dimension: spec = spec[:, torch.randperm(spec.shape[-2]), :]
    return spec

def get_specaugment_config_from_args(args):
    spec_augment_args = {k.replace('spec_augment_', ''):v for k,v in args.__dict__.items() if k.startswith('spec_augment')}
    spec_augment_config = {
        'n_time_masks': spec_augment_args.get('n_time_masks', 0),
        'n_freq_masks': spec_augment_args.get('n_freq_masks', 0),
        'freq_mask_param': spec_augment_args.get('freq_mask_param', 42),
        'time_mask_param': spec_augment_args.get('time_mask_param', -1),
        'min_p': spec_augment_args.get('min_p', 0.05),
        'zero_masking': spec_augment_args.get('zero_masking', False),
    }
    return spec_augment_config

def get_frame_shuffle_config_from_args(args):
    frame_shuffle_args = {k.replace('frame_shuffle_', ''):v for k,v in args.__dict__.items() if k.startswith('frame_shuffle')}
    frame_shuffle_config = {
        'time_dimension': frame_shuffle_args.get('time_dimension', False),
        'freq_dimension': frame_shuffle_args.get('freq_dimension', False),
    }
    return frame_shuffle_config

def get_lr_args_from_args(args):
    lr_args = {k.replace('optim_', ''):v for k,v in args.__dict__.items() if k.startswith('optim_')}
    lr_args['lr'] = lr_args.get('lr', 9e-5)
    return lr_args


def prepare_chunks(spec, seq_len, overlap):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False

    if spec_n <= seq_len:
        return {0: spec}, [0]

    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())


def bitfit(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, FusedLayerNorm) or isinstance(module, torch.nn.LayerNorm):
            module.bias.requires_grad = True
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.requires_grad = True
        if isinstance(module, BatchRenorm1d):
            module.bias.requires_grad = True

    return model

def AWMC(
        args,
        model:nn.Module,
        spec:torch.Tensor,
        seq_len:int,
        overlap:int,
        tokenizer,
        use_tqdm:bool=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        optimizer_state:dict=None,  
        beam_search_fn:Callable=None,
        return_params:bool=False,

    ):
    
    assert beam_search_fn is None, 'Beam search function not implemented for AWMC'
    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    frame_shuffle_args = get_frame_shuffle_config_from_args(args)
    
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]

    if args.__dict__.get('bitfit', False):
        model = bitfit(model)

    model.train()
    ema_leader_model = ExponentialMovingAverage(model.parameters(), decay=args.__dict__.get('ema_decay', 0.999))
    ema_leader_model.update()
    ema_anchor_model = ExponentialMovingAverage(model.parameters(), decay=1.0) # no decay
    ema_anchor_model.update()

    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    
    optimizer = optim(model.parameters(), **lr_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)
    augmentation = SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    assert args.config['training'].get("max_seq_len", 0) == 0, 'caching is not used anymore'
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    epochs = args.__dict__.get('epochs', 1)
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    training_keys = list(training_data.keys())
    pbar = tqdm(training_keys) if use_tqdm else training_keys

    print_runtimes = args.__dict__.get('print_runtimes', False)
    if print_runtimes: print('Spectrogram length:', spec_n)
    stime = time.time()

    model_outputs = {}
    model.eval()
    for i in pbar:
        label_bank = [None, None]
        for j in range(epochs):
            audio_chunk = training_data[i].clone().to(model.device)
            if j == 0:
                with ema_anchor_model.average_parameters() as anchor_params, torch.no_grad() as g:
                    out = model(audio_signal = audio_chunk)
                    pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
                    #print(f'Pseudo targets: {pseudo_targets}')
                    pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device)
                    label_bank[0] = pseudo_targets.transpose(0, 1) 
                    
            with ema_leader_model.average_parameters() as leader_params, torch.no_grad() as g:
                out = model(audio_signal = audio_chunk)
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
                #print(f'Pseudo targets: {pseudo_targets}')
                pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device)
              
                label_bank[1] = pseudo_targets.transpose(0, 1)
              
            audio_chunk = augmentation(audio_chunk)
            audio_chunk = frame_shuffle(audio_chunk, **frame_shuffle_args)

            out = model(audio_signal = audio_chunk)
            predictions = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
            print(f'Noisy Predictions: {predictions}')
            predictions = torch.LongTensor(tokenizer.encode(predictions)).unsqueeze(0).to(model.device)
            
            labels = [el for el in label_bank if el.shape[0] > 0]
   
            label_bank_lengths = torch.LongTensor([el.shape[0] for el in labels]).to(model.device)

            if len(labels) == 0:
                labels = [torch.LongTensor([[]]).T.to(model.device)]
                label_bank_lengths = torch.LongTensor([0]).to(model.device)
      
            labels = torch.nn.utils.rnn.pad_sequence(sequences=labels, batch_first=False, padding_value=0)
      
            labels = labels.squeeze(2).transpose(0, 1)
            N, B = out['final_posteriors'].shape[1], out['final_posteriors'].shape[0]
            total_tokens_in_loss = N * B * 2

            #print(label_bank)

            loss = ctc_loss_fn(
                out['final_posteriors'].repeat(label_bank_lengths.shape[0], 1, 1).transpose(0, 1),
                targets = labels, 
                input_lengths = torch.LongTensor([N] * labels.shape[0]).to(model.device),
                target_lengths = label_bank_lengths,
            ) / total_tokens_in_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            ema_leader_model.update()
            #ema_anchor_model.update()
            if j ==  epochs - 1:
                audio_chunk = training_data[i].clone().to(model.device)
                with torch.no_grad(): out = model(audio_signal = audio_chunk)
                logits = out['final_posteriors'][0].detach().cpu()
                logits = torch.exp(logits) # convert to prob
                
                ds_len = logits.shape[-2]
                ratio = audio_chunk.shape[-1] / ds_len
                overlap_ds = int(overlap / ratio)
          
                model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

    etime = time.time()
    if print_runtimes: print(f'Runtime: {etime - stime}')
            
    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 

    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    logits = torch.log(logits) # convert to log 
    
    
    if return_params:
        updated_model_params = list(model.parameters())
        updated_model_params = [p.clone().detach().cpu() for p in updated_model_params]

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)

    return logits.squeeze(0).numpy() if not return_params else (logits.squeeze(0).numpy(), updated_model_params)           
                    

def add_random_noise(spec, noise_factor):
    if noise_factor == 0: return spec
    noise = torch.normal(0, std=spec.std(), size=spec.shape).to(spec.device)
    return spec + noise * noise_factor

def cutout(spec, seq_len, cutout_val=' -r $REPEAT', num_rectangles=5, max_width=100, max_height=10):
    '''
    cutout_val: 'mean', 'mean_recording', 'zero'
    assumes a batch size of 1 (rearange to (F, B*N) if batch size > 1)
    '''
    if num_rectangles == 0: return spec

    spec_n = spec.shape[-1]
    ratio = spec_n / seq_len
    num_rectangles = int(num_rectangles * ratio) # if this spectrogram is shorter than the sequence lentgth used for tuning reduce the number of rectangles

    widths = torch.randint(1, max_width, (num_rectangles,))
    heights = torch.randint(1, max_height, (num_rectangles,))
    start_positions_x = torch.randint(0, spec.shape[-1], (num_rectangles,))
    end_positions_x = (start_positions_x + widths).clamp(max=spec.shape[-1])
    start_positions_y = torch.randint(0, spec.shape[-2], (num_rectangles,))
    end_positions_y = (start_positions_y + heights).clamp(max=spec.shape[-2])
    
    if cutout_val == 'mean_recording':
        mask_value = spec.mean()
    elif cutout_val == 'mean':
        mask_values = []
        for i in range(num_rectangles):
            mask_values.append(spec[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]].mean())

    for i in range(num_rectangles):
        #print(start_positions_x[i], end_positions_x[i], start_positions_y[i], end_positions_y[i])
        if cutout_val == 'mean':
            spec[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]] = mask_values[i]
        elif cutout_val == 'mean_recording':
            spec[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]] = mask_value
        elif cutout_val == 'zero':
            spec[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]].zero_()
    return spec

def get_cutout_params_from_args(args, seq_len):
    cutout_args = {k.replace('cutout_', ''):v for k,v in args.__dict__.items() if k.startswith('cutout')}
    cutout_config = {
        'seq_len': seq_len, 
        'cutout_val': cutout_args.get('value', 'mean'),
        'num_rectangles': cutout_args.get('num_rectangles', 0),
        'max_width': cutout_args.get('max_width', 100),
        'max_height': cutout_args.get('max_height', 10),
    }
    return cutout_config

def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        optimizer_state:dict=None,
        beam_search_fn:Callable=None,
        return_params:bool=False,
    ):
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    spec_augment_config = get_specaugment_config_from_args(args)
    random_noise = args.__dict__.get('random_noise', 0.0)

    lr_args = get_lr_args_from_args(args)
    frame_shuffle_args = get_frame_shuffle_config_from_args(args)

    
    cutout_args = get_cutout_params_from_args(args, seq_len)
    print(spec_augment_config, lr_args, frame_shuffle_args, cutout_args)
    num_negatives = 1
    


    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]
 
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    
    optimizer = optim(model.parameters(), **lr_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)
    augmentation = SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    assert args.config['training'].get("max_seq_len", 0) == 0, 'caching is not used anymore'
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))

    epochs = args.__dict__.get('epochs', 1)
    shuffle = args.__dict__.get('shuffle', False)
    online = args.__dict__.get('online', False)
    beams = args.__dict__.get('lm_tta_beams', 3)
    epochs = 1 if online else epochs
    shuffle = False if online else shuffle
    model_outputs = {}

    print_runtimes = args.__dict__.get('print_runtimes', False)

    if print_runtimes: print('Spectrogram length:', spec_n)

    entropy = []
    model.eval() # don't update batchrenorm
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {epochs}')
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if shuffle else training_keys

        epochs_stime = time.time()
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1) # [B, C, T]
            print(audio_chunk[:num_negatives].shape)
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies
            audio_chunk[:num_negatives] = frame_shuffle(audio_chunk[:num_negatives], **frame_shuffle_args)
            audio_chunk[:num_negatives] = add_random_noise(audio_chunk[:num_negatives], noise_factor = random_noise)
            audio_chunk[:num_negatives] = cutout(audio_chunk[:num_negatives], **cutout_args)
           

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)
            # #entrop = torch.distributions.Categorical(probs = out['final_posteriors'][-1].detach().cpu().exp().mean()).entropy()
            # entrop = out['final_posteriors'][-1].detach().cpu().exp().max(-1).values
            # print(f'Entropy: {entrop.mean()}')
            # entropy.append(entrop.mean().item())
            # plt.plot(entropy)
            # plt.savefig('entropy.png')

            if beam_search_fn is None or beams == 0: 
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            else:
                beam_search = beam_search_fn(log_probs = out['final_posteriors'][-1].detach().cpu(), beam_width = beams)
                beam_search.run_search(use_tqdm = True)
                pseudo_targets = beam_search.return_text(idx = 0)

            noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
            print(f'Pseudo targets: {pseudo_targets}')
            print(f'Noisy predictions: {noisy_predictions}')
            print('\n--\n')
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B
 
            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8) # add clip value to args
            optimizer.step()

            if online:
                logits = out['final_posteriors'][-1].detach().cpu() 
                logits = torch.exp(logits) # convert to prob
                ds_len = logits.shape[-2]
                ratio = u_len / ds_len
                overlap_ds = int(overlap / ratio)
                model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
        epochs_etime = time.time()
        if print_runtimes: print(f'Epoch runtime: {epochs_etime - epochs_stime}')

        
    if not online:
        model.eval()
        training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
        final_pass_stime = time.time()
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            with torch.no_grad(): out = model(audio_signal = audio_chunk)
            logits = out['final_posteriors'][0].detach().cpu()
            logits = torch.exp(logits) # convert to prob
            ds_len = logits.shape[-2]
            ratio = u_len / ds_len
            overlap_ds = int(overlap / ratio)
            model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
        final_pass_etime = time.time()
        if print_runtimes: print(f'Final pass runtime: {final_pass_etime - final_pass_stime}')
        model.train()

           
    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 

    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    logits = torch.log(logits) # convert to log 

    if return_params:
        updated_model_params = list(model.parameters())
        updated_model_params = [p.clone().detach().cpu() for p in updated_model_params]

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)


    return logits.squeeze(0).numpy() if not return_params else (logits.squeeze(0).numpy(), updated_model_params)


dynamic_eval = dynamic_eval_ctc_loss




    # parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to checkpoint')
    # parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    # parser.add_argument('-seq', '--seq_len', type=int, default=16384, help='-1 to use setting from config in checkpoint file')
    # parser.add_argument('-o', '--overlap', type=int, default=14336, help='-1 to use setting from config in checkpoint file')
    # parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    # parser.add_argument('-log', '--log', type=str, default='')
    # parser.add_argument('-ds', '--dont_shuffle', action='store_true', help='dont shuffle')
    # #parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    # parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    # parser.add_argument('-dfa', '--disable_flash_attention', action='store_true', help='disable flash attention')
    # parser.add_argument('-beamsearch', '--beamsearch', action='store_true', help='use beam search')
    # parser.add_argument('-kwargs', '--kwargs', nargs='+', help='kwargs')
    # parser.add_argument('-awmc', '--awmc', action='store_true', help='Use AWMC method from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10389640&tag=1 instead of dynamic eval')
