import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')

    parser.add_argument('--long_len', type=int, default=0, help='long len like 64, 128, 256')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    return args

def replace_value_in_dict(d, old_value, new_value):
    if isinstance(d, dict):
        return {k: replace_value_in_dict(v, old_value, new_value) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_value_in_dict(v, old_value, new_value) for v in d]
    else:
        return new_value if d == old_value else d



def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    ### change the configs and args    
    import os
    config = replace_value_in_dict(config, 24, int(args.long_len))  ## to change the default 24 length of time to 64,128,256


    args.save_dir = Path(os.environ.get('results_folder'))
    config['solver']['ema']['decay'] = 0.995  ## hucfg912


    logger = Logger(args)
    logger.save_config(config)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = instantiate_from_config(config['model']).to(device)

    
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    # import ipdb; ipdb.set_trace()
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
    # import ipdb; ipdb.set_trace()

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}_{args.long_len}.npy'), samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}_{args.long_len}.npy'), samples)

if __name__ == '__main__':
    main()
