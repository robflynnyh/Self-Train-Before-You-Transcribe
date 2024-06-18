# Self-Train-Before-You-Transcribe
Code for Interspeech 2024 paper: Self-Train Before You Transcribe
![figure and algorithm 1 in paper](https://github.com/robflynnyh/Self-Train-Before-You-Transcribe/blob/main/nsti.png?raw=true)

- Only basic code is provided at the moment will be updated in future with a more complete version. In the mean time create an issue or email me if you have any problems.
- code.py contains the functions that were used for the primary experiments in the paper. Specifically the dynamic_eval_ctc_loss function is the main method that is presented. 
- ./Earnings22_NST_finetune/train.py contains code that was used to finetune one of our models on Earnings22 using the Noisy Student Training (NST) method from https://arxiv.org/pdf/2106.08922.pdf 
- For this work models from https://github.com/robflynnyh/long-context-asr (lcasr) are used and many of the functions depend on this library.
- This link to the pre-trained checkpoint used in the this work is available here: https://huggingface.co/rjflynn2/lcasr-6L-768D-6H-RB-10M/tree/main/n_seq_sched_16384_rp_1 
