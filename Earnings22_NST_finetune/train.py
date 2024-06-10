'''
Finetuning on Earnings-22 using momentum based noisy student teacher training.
'''

import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
import traceback
from lcasr.utils.dataloading import chunk_spectogram, reset_seen_ids, load_sample, collate_fn
from lcasr.utils.hooks import add_debug_backwards_hooks
from lcasr.utils.scheduling import CosineLRScheduler, SequenceWarmupManager
from lcasr.utils.helpers import exists
from lcasr.utils.general import load_model, save_model, load_checkpoint, load_optimizer, find_latest_checkpoint
from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder

from einops import rearrange
import numpy as np
import os, wandb, resource
from contextlib import nullcontext
from functools import partial
import sentencepiece as spm
from torch.cuda.amp import GradScaler
from torch import autocast
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings,random
random.seed(1234)
import pandas as pd
from torch_ema import ExponentialMovingAverage
from run_eval import EvalRunner


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            pairs:Dict[str, Dict[str, str]],
            batch_size:int = 8,
            subgroup_shuffle_size:int = 2000,
            skip_to:int = 0, # deprecated
            random_seed:int = 1234,
            seen_ids:List[str] = [], # remove ids from dataset (i.e already trained on)
        ):
        self.batch_size = batch_size
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.random_seed = random_seed
  
        self.pairs = pd.DataFrame(list(pairs.values()))
        self.pairs['id'] = list(pairs.keys())
        # remove ids
        self.pairs = self.pairs[~self.pairs['id'].isin(seen_ids)]

        # sort pairs by duration
        self.pairs = self.pairs.sort_values(by='duration')
        self.pairs = self.pairs.reset_index(drop=True) # reset index drop means old index is not added as a column

        self.create_batches()
        # trim to skip_to
        self.pairs = self.pairs.iloc[skip_to:].reset_index(drop=True) # deprecated in favour of seen_ids !!
       

    def create_batches(self):
        np.random.seed(self.random_seed)
        indices = np.arange(len(self))
        indices = [np.random.permutation(indices[i:i+self.subgroup_shuffle_size]) for i in range(0, len(indices), self.subgroup_shuffle_size)]
        indices = np.concatenate(indices)
        indices = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        np.random.shuffle(indices)
        indices = np.concatenate(indices)
        self.pairs = self.pairs.iloc[indices].reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio = torch.load(self.pairs['spectrogram'][idx])
        id = self.pairs['id'][idx]
        audio = rearrange(audio, '() f t -> t f')
        return audio, None, id # none is text which we don't have due to NST

class SimpleDataloader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        pairs:Dict[str, Dict[str, str]], 
        tokenizer:spm.SentencePieceProcessor, 
        skip_to:int = 0,
        batch_size:int = 5,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
        num_workers:int = 0,
        pin_memory:bool = False,
        prefetch:int = None,
        random_seed=1234,
        subgroup_shuffle_size:int = 2000,
        seen_ids:List[str] = [],
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.skip_to = skip_to

        dataset = SimpleDataset(
                    pairs, 
                    batch_size = batch_size,
                    skip_to = skip_to, 
                    subgroup_shuffle_size = subgroup_shuffle_size,
                    random_seed = random_seed,
                    seen_ids = seen_ids,
        )
        super().__init__(
                dataset = dataset,
                batch_size = batch_size, 
                shuffle = False, 
                num_workers = num_workers, 
                pin_memory = pin_memory, 
                collate_fn = collate_fn(),
                prefetch_factor = prefetch if num_workers > 0 else None,
            )
            

class VariableBatchSimpleDataloader():
    def __init__(
        self, 
        pairs:Dict[str, Dict[str, str]], 
        tokenizer:spm.SentencePieceProcessor, 
        skip_to:int = 0,
        batch_size:int = 5,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
        num_workers:int = 0,
        pin_memory:bool = False,
        prefetch:int = None,
        random_seed=1234,
        subgroup_shuffle_size:int = 2000,
        seen_ids:List[str] = []
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.pairs = pairs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.random_seed = random_seed

        self.dataloader = SimpleDataloader(
            pairs = pairs,
            tokenizer = tokenizer,
            skip_to = skip_to,
            batch_size = batch_size,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            num_workers = num_workers,
            pin_memory = pin_memory,
            prefetch = prefetch,
            subgroup_shuffle_size = subgroup_shuffle_size,
            random_seed = random_seed,
            seen_ids = seen_ids,
        )

    def update(
            self, 
            batch_size:int, 
            seen_ids:List[str]=[],
            random_seed:int='same'
        ):
        self.batch_size = batch_size
        self.dataloader = SimpleDataloader(
            pairs = self.pairs,
            tokenizer = self.tokenizer,
            batch_size = batch_size,
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            prefetch = self.prefetch,
            random_seed = self.random_seed if random_seed == 'same' else random_seed,
            seen_ids = seen_ids,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def total_recordings(self):
        return len(self.pairs.keys())

    def __len__(self):
        return len(self.dataloader) 

def blank_p(logits, tokenizer):
    lset = logits.detach().cpu()
    if torch.rand(1) < 0.05: # print 5 percent of the time
        print(tokenizer.decode([el for el in lset[0].argmax(dim=-1).tolist() if el != lset.shape[-1]-1]))
    lset = rearrange(lset, 'b n v -> (b n)   v')
    lset_max = lset.argmax(dim=-1)
    lset_max = lset_max[lset_max == (lset.shape[-1]-1)]
    blank_p = lset_max.shape[0] / lset.shape[0]
    return blank_p

def backwards_pass(
        model:SCConformerXL,
        clip_value:float,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler,
        scaler:GradScaler,  
        ema:ExponentialMovingAverage,
    ):
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) if clip_value > 0 else None
    scaler.step(optimizer)
    scaler.update()
    ema.update()
    optimizer.zero_grad()

    if scheduler.is_warmup:
        scheduler.step()


def apply_augmentation(audio, lengths, augmentation, epoch, start_augment_after_n_epochs, is_warmup):
    if start_augment_after_n_epochs == -1 or epoch < start_augment_after_n_epochs or not exists(augmentation) or is_warmup:
        return audio
    else:
        return augmentation(audio, lengths)
    
def get_dtype(dtype:str) -> torch.dtype:
    if dtype == 'bfloat16':
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16
    elif dtype == 'float32':
        return torch.float32
    else:
        raise ValueError(f'invalid dtype: {dtype}')

def NST(model, ema, augmentation, ctc_decoder, ctc_loss_fn, batch, tokenizer):
    #print(batch['audio_signal'].shape)
   
    with torch.no_grad(), ema.average_parameters():
        ema_outs = model(**batch)
        ema_probs = ema_outs['final_posteriors']
    ema_labels = [torch.LongTensor(ctc_decoder(el)) for el in ema_probs]
    #print(tokenizer.decode(ema_labels[0].tolist()), "!")
    ema_label_lengths = torch.LongTensor([len(el) for el in ema_labels])
    ema_labels = torch.nn.utils.rnn.pad_sequence(ema_labels, batch_first=True, padding_value=0)
    ema_labels = ema_labels.to(batch['audio_signal'].device)

    batch['audio_signal'] = apply_augmentation(audio=batch['audio_signal'], lengths=batch['length'], augmentation=augmentation, start_augment_after_n_epochs=0, epoch=0, is_warmup=False)

    out = model(**batch)
    probs = out['final_posteriors']

    loss = ctc_loss_fn(probs.transpose(0,1), ema_labels, out['length'], ema_label_lengths).sum() # !!
    return loss, probs
    


def train(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer,
        scheduler:CosineLRScheduler,
        sequence_scheduler:SequenceWarmupManager,
        device:torch.device,
        ema:ExponentialMovingAverage,
        step:int = 0,
        seen_ids:List[str] = [],
        epoch:int = 0,
        augmentation:SpecAugment = None,
    ):

    scaler, clip_value, wandb_config, dtype, rlimit = GradScaler(), args.config['training'].get('clip_value', 0.8), args.config['wandb'], get_dtype(args.config['training'].get('dtype', 'bfloat16')), resource.getrlimit(resource.RLIMIT_NOFILE)
    random.seed(args.config['training'].get('random_seed', 12345))
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    model.train()
    ema.update() # otherwise weights of ema model are randomly initialized
    
    model_dtype, ctc_loss_fn = next(model.parameters()).dtype, torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    backprop_every, backwards_every = args.config['training']['backprop_every'], args.config['training'].get('backwards_every', 1)

    assert backprop_every >= backwards_every, f'backprop_every ({backprop_every}) must be >= backwards_every ({backwards_every})'
    
    batch_size, max_cache_length = args.config['training']['batch_size'], args.config['training'].get('max_cache_length', 0)
    cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)

    ctc_decoder = GreedyCTCDecoder(blank_id = model.decoder.num_classes-1)
    tokenizer = dataloader.tokenizer
    chunk_size, chunk_overlap = args.config.audio_chunking['size'], 0 # previously args.config.audio_chunking['overlap'] though this is not used anymore

    if exists(sequence_scheduler): chunk_size, batch_size = sequence_scheduler.cur_sequence_length, sequence_scheduler.cur_batch_size
        
    pad_id = tokenizer.pad_id()
    last_podcast, cur_podcast, podcasts_since_last_save = step, step, 0
    max_epochs = args.config['training'].get('max_epochs', 1)

    i, finished, dataloader_iter, total_recordings = -1, False, iter(dataloader), dataloader.total_recordings() * max_epochs
    pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')
    
    evalrunner = EvalRunner(tokenizer=tokenizer, split='dev')
    #initial_wer = evalrunner.run_eval(model = model, device=device, seq_len=chunk_size, overlap=int(chunk_size*0.875))
    wandb.log({'wer':0.2388304862023653}) if wandb_config['use'] else None

    while not finished:#################
        try:
            batch, i = next(dataloader_iter), i + 1
            pbar.update(1) if i > 0 else None
        except StopIteration:
            wer = evalrunner.run_eval(model = model, device=device, seq_len=chunk_size, overlap=int(chunk_size*0.875))
            wandb.log({'wer':wer}) if wandb_config['use'] else None
            epoch += 1
            seen_ids = reset_seen_ids(seen_ids = seen_ids, epoch = epoch - 1)
            # save model
            torch.cuda.empty_cache() 
            save_model(
                model = model, 
                optimizer = optimizer, 
                scheduler = scheduler, 
                podcast_step = cur_podcast, 
                config = args.config,
                sequence_scheduler = sequence_scheduler,
                seen_ids = seen_ids,
                epoch = epoch,
                other = {'ema_weights':ema.state_dict()},
            )
            podcasts_since_last_save = 0
            if epoch >= max_epochs:
                finished = True
            else:
                dataloader.update(
                    batch_size = dataloader.batch_size, 
                    seen_ids = seen_ids,
                    random_seed = random.randint(0, 10000),
                )
                dataloader_iter = iter(dataloader)
                pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')
            continue
        ################################

        audio, audio_lengths, txt, ids = batch
        seen_ids.extend(ids)
        cur_batch_size = audio.shape[0]

        ###############################
        cur_podcast += audio.shape[0]
        podcasts_since_last_save += (cur_podcast - last_podcast)
        last_podcast = cur_podcast
        ###############################
        
        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)

        del audio
        backwards_every_loss, steps_since_backwards = 0.0, 0
        chunks, culm_lengths_audio, nans_in_a_row = [], torch.zeros_like(audio_lengths), 0

        ################################
        for ix, el in enumerate(audio_chunks_):

            remove_mask = ~(culm_lengths_audio > audio_lengths)
            cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
            cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths[remove_mask] - chunk_overlap).clamp(0)
          
            chunks.append({
                'audio':cur_chunks,
                'audio_lengths':cur_lengths,
                'selection_mask':remove_mask,
                'cur_culm_lengths':cur_culm_lengths,
            })
            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                scheduler.set_cosine_schedule(total_recordings=total_recordings, cur_podcast=cur_podcast)
        ################################
        # shuffle chunks
        random.shuffle(chunks)
        try:
            for ix, chunk_json in enumerate(chunks):
                print(f'chunk {ix}/{len(chunks)}')
               
                audio, a_lengths = chunk_json['audio'], chunk_json['audio_lengths']
                audio, a_lengths = audio.to(device, dtype=model_dtype), a_lengths.to(device)

                with autocast(device.type, dtype=dtype) if torch.cuda.is_available() else nullcontext():
                    loss, probs = NST(
                        model = model,
                        ema = ema,
                        augmentation = augmentation,
                        ctc_decoder = ctc_decoder,
                        ctc_loss_fn = ctc_loss_fn,
                        batch = {"audio_signal":audio, "length":a_lengths},
                        tokenizer=tokenizer,
                    )
                    
                blank_prob = blank_p(probs.detach(), dataloader.tokenizer)
                # check for nan in loss
                if torch.isnan(loss):
                    print('OH NO! NAN IN LOSS, SKIPPING') # TODO: set kv cache to None here
                    wandb.log({'nan':True}) if wandb_config['use'] else None
                    optimizer.zero_grad() # clear gradients
                    nans_in_a_row += 1
                    if nans_in_a_row > 100:
                        print('100 NANS in a row, exiting......')
                        exit()
                    continue
                else:
                    nans_in_a_row = 0


                cur_loss += loss
                backwards_every_loss += loss
                steps_since_backwards += 1
                cur_tokens_in_loss += (sum(a_lengths)) # total number of acoustic frames in batch

                if (ix+1) % backwards_every == 0 or (ix+1) == len(chunks):
                    scaler.scale(((backwards_every_loss) / (chunk_size*batch_size)*steps_since_backwards) * 100).backward() # divide by chunk*batch_size constant to weight smaller batches less
                    steps_since_backwards, backwards_every_loss = 0, 0

                if (ix+1) % backprop_every == 0 or (ix+1) == len(chunks): 
                    full_loss = (cur_loss / cur_tokens_in_loss) * 100
                    loss_to_log = full_loss.item()
                    print(f'loss: {full_loss}')
                    
                    backwards_pass(
                        model = model,
                        clip_value = clip_value,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        scaler = scaler,
                        ema = ema,
                    )
                    learning_rate = scheduler.get_last_lr()[0]
                 

                    if wandb_config['use']:
                        wandb.log({
                            'loss': loss_to_log,
                            'blank_p': blank_prob,
                            'learning_rate': learning_rate,
                            'sequence_length': chunk_size,
                            'batch_size': batch_size,
                            'epoch': epoch,
                        })
                    
                    cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)
                
        except RuntimeError as e: 
            if 'an illegal memory access was encountered' in str(e): 
                print(e,'\n --- skipping batch ---')
                continue
            else:
                print(traceback.format_exc()) 
                raise e

        if not scheduler.is_warmup: # step every batch
            scheduler.step(epoch = cur_podcast)

        if exists(sequence_scheduler):
            to_update, new_seq_len, new_bs = sequence_scheduler.step(steps = cur_batch_size)
            if to_update:
                args.config['audio_chunking']['size'] = new_seq_len
                chunk_size = new_seq_len
                batch_size = new_bs
                dataloader.update(
                    batch_size = batch_size,
                    seen_ids = seen_ids,
                )
                if args.config['model']['use_rotary'] and args.config['sequence_scheduler'].get('interpolate_rotary', False):
                    model.rotary_pos_emb.rotary_interpolation_factor = model.rotary_pos_emb.rotary_interpolation_factor * sequence_scheduler.increase_by_multiplier
                dataloader_iter = iter(dataloader)
                pbar.total = len(dataloader) # update total of tqdm
                
        del chunks
        

    return model
            
            


def main(args):
    args.config_path = args.config
    args.config = OmegaConf.load(args.config)

    checkpoint_dir = args.config['checkpointing']['dir']
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir); print(f'created checkpoint dir: {checkpoint_dir}')

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    # set random seed for initialization
    torch.manual_seed(12345), torch.cuda.manual_seed(12345)
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    paired_data = lcasr.utils.audio_tools.load_json(args.config['data']['path'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb_config = args.config['wandb']
    if wandb_config['use']:
        project_name, w_id = wandb_config['project_name'], wandb_config['id']
        run_name = None if 'name' not in wandb_config else wandb_config['name']
        config_dict = OmegaConf.to_container(args.config, resolve=True)
        wandb.init(project=project_name, config=config_dict, name=run_name) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=config_dict, allow_val_change=True)
        wandb.watch(model, log="all") # sometimes this causes a crash ):
        wandb.config.update({'total_params': tparams}, allow_val_change=True)
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')
        args.config['wandb']['id'] = wandb.run.id # add wandb config to args.config
        if wandb_config.get('update_config_with_wandb_id', False): OmegaConf.save(config=args.config, f=args.config_path)

    model = model.to(device)
    optimizer, scheduler = load_optimizer(args.config, model)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.config['training'].get('ema_decay', 0.999))


    sequence_scheduler = None
    if 'sequence_scheduler' in args.config:
        sequence_scheduler = SequenceWarmupManager(
            initial_batch_size = args.config['training']['batch_size'],
            initial_sequence_length = args.config['audio_chunking']['size'],
            **args.config['sequence_scheduler']
        )

    seen_ids, step, epoch = load_checkpoint(
        args = args, 
        model = model, 
        optimizer = optimizer if not args.reset_optim else None,
        scheduler = scheduler, 
        sequence_scheduler = sequence_scheduler,
        path = args.config['checkpointing']['dir'],
        device = device,
        other = [(ema, 'ema_weights')]
    )
    if args.reset_step:
        seen_ids, step, epoch = [], 0, 0 

    print(f'Starting from podcast: {len(seen_ids)}')
    random_seed = args.config['training'].get('random_seed', 1234)
    

    # skip data up to step
    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.config['training']['batch_size'],
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
        num_workers = args.num_workers,
        pin_memory = args.pin_memory,
        prefetch = args.prefetch_factor,
        seen_ids = seen_ids,
        random_seed = random_seed,
    )

    # None if start_spec_augment_after_n_epochs == -1 or epoch < start_spec_augment_after_n_epochs else 
    augmentation = SpecAugment(**args.config['spec_augment']) if 'spec_augment' in args.config else None
    assert exists(augmentation), 'must have spec augment in config for noisy student teacher training'

    if args.debug_hooks:
        assert wandb_config['use'], 'must have wandb enabled when - arg.debug_hooks ==  True - to log debug hooks outputs'
        logger = partial(wandb.log, commit=False)
        add_debug_backwards_hooks(model = model, logger = logger)
    
    if sequence_scheduler and dataloader.batch_size != sequence_scheduler.cur_batch_size:
        print('WARNING: dataloader batch size does not match sequence scheduler batch size, updating dataloader batch size')
        dataloader.update(batch_size = sequence_scheduler.cur_batch_size, seen_ids = seen_ids)

    final_model = train(
        args = args, 
        model = model, 
        dataloader = dataloader, 
        optimizer = optimizer, 
        scheduler = scheduler,
        sequence_scheduler = sequence_scheduler, 
        device = device, 
        seen_ids = seen_ids,
        step = step,
        augmentation = augmentation,
        epoch = epoch,
        ema = ema
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-rm_sched', '--remove_scheduler', action='store_true', help='remove scheduler from checkpoint')
    parser.add_argument('-reset_step', '--reset_step', action='store_true', help='reset step to 0')
    parser.add_argument('-reset_optim', '--reset_optim', action='store_true', help='reset optimizer to default')
    parser.add_argument('-anomaly', '--anomaly', action='store_true', help='turn on anomaly detection')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-pin_memory', '--pin_memory', action='store_true', help='pin memory for dataloader')
    parser.add_argument('-prefetch', '--prefetch_factor', type=int, default=1, help='prefetch factor for dataloader')

    parser.add_argument('-debug_hooks', '--debug_hooks', action='store_true', help='add hooks to log gradient/activation info')

    args = parser.parse_args()


    main(args)
      
