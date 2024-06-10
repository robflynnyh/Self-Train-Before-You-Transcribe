# Self-Train-Before-You-Transcribe
Code for Interspeech 2024 paper: Self-Train Before You Transcribe

- Only basic code is provided at the moment will be updated in future with a more complete version. In the mean time create an issue or email me if you have any problems.
- code.py contains the functions that were used for the primary experiments in the paper. Specifically the dynamic_eval_ctc_loss function is the main method that is presented. 
- ./Earnings22_NST_finetune/train.py contains code that was used to finetune one of our models on Earnings22 using the Noisy Student Training (NST) method from https://arxiv.org/pdf/2106.08922.pdf 
- For this work models from https://github.com/robflynnyh/long-context-asr (lcasr) are used and many of the functions depend on this library. 