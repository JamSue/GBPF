# GBPF

## Environment Setup

To ensure reproducibility of the experiments, use the following Conda environment configuration file:

1. Clone or download this repository.
2. In the project directory, create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate myenv
   ```

   torch may be installed by:  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

## Dataset processing

    Our datasets are all in directory './dataset', saved as csv files
    format as "label", "sentence"

    example from AGNEWS:
    "2","EU warns Italy on Alitalia stake The Italian state must give up its majority stake in troubled Alitalia when the airline sells new shares, the EU warns."

    download the dataset from following url:
                https://www.dropbox.com/scl/fo/684zdkddk6o6cpbvlwyh2/APkd4LCQ6QmQ3rpLyv0kgdU?rlkey=udlqiesi2t6xser6ahox6citl&st=a7ghrnnx&dl=0
## Model Download
    for classical model, you don't need to download models, just import a package by `from torch import nn`;

    for Bert Serise, down Bert, XLNet and RoBERTa from huggingFace or by script.
## Training Command

For classical models:
`python train.py --model LSTM --dataset AGNEWS --batch_size 128 --augment_num 5`

For Bert Series:
`python trainBert.py --model Bert --dataset AGNEWS --batch 64 --augnent_num 5` (single GPU or with GPU)
`python  trainBertDP.py --model Bert --dataset AGNEWS --batch 64 --augnent_num 5` （muti-GPU parallel）

during training we will obtain models saved in './checkpoints', and Granular_Ball Space will save at './gb_data'

## Defend Process

For classical models:
`python attack.py --model LSTM --dataset AGNEWS --attack_method PWWS  --pretrained_model_path (target model pth file path) --ball_path (your GBS saved path，such as: `gb_data/AGNEWS_LSTM_ballData_.npy`) --k 20`
For Bert Series:
`python textattack.py --model Bert --dataset AGNEWS --attack_method PWWS --pretrained_model_path (target model pth file path) --ball_path (your GBS saved path，such as: `gb_data/AGNEWS_Bert_ballData_.npy`) --k 20 `
    for example:
    `python textattack.py --model Bert --dataset AGNEWS --attack_method PWWS --pretrained_model_path model\model_Bert_AGNEWS_best_93.75.pth --ball_path gb_data\AGNEWS_Bert_ballData_droupout0.3.npy --k 20 `

# parameters

experiment setting are in file `config.py`

# ablation experiment

## mertic leaning
    for classical models:
        nohup python train_ablation_metricLearning.py --model LSTM --dataset YAHOO  --batch_size 128 --augment_num 5 &
    
    for bert series:
        nohup python train_ablation_metricLearning——Bert.py --model Bert --dataset AGNEWS  --batch_size 128 --augment_num 5 &
## data augment