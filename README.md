# GBPF

## Environment Setup

To ensure reproducibility of the experiments, use the following Conda environment configuration file:

1. Clone or download this repository.
2. In the project directory, create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate myenv
   ```

## 1. dataset processing

    Our datasets are all in directory './dataset', saved as csv files
    format as "label", "sentence"

    example from AGNEWS:
    "2","EU warns Italy on Alitalia stake The Italian state must give up its majority stake in troubled Alitalia when the airline sells new shares, the EU warns."

## 2. training command

For classical models:
`python train.py --model LSTM --dataset AGNEWS --batch_size 128 --augment_num 5`

For Bert Series:
`python trainBert.py --model Bert --dataset AGNEWS --batch 64 --augnent_num 5` (single GPU or with GPU)
`python  trainBertDP.py --model Bert --dataset AGNEWS --batch 64 --augnent_num 5` （muti-GPU parallel）

during training we will obtain models saved in './checkpoints', and Granular_Ball Space will save at './gb_data'

## 3. defend process

For classical models:
`python attack.py --model LSTM --dataset AGNEWS --attack_method PWWS  --pretrained_model_path (target model pth file path) --k 20`
For Bert Series:
`python textattack.py --model Bert --dataset AGNEWS --attack_method PWWS --pretrained_model_path (target model pth file path) --k 20 `

# parameters

experiment setting are in file `config.py`
