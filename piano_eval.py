import argparse
import os

import torch
from hamer.configs import dataset_eval_config
from hamer.datasets import create_dataset
from hamer.utils import Evaluator, recursive_to
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on Fur Elise dataset')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--results_folder', type=str, default='results', help='Path to results folder.')

    args = parser.parse_args()

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    

    # Create dataset and data loader
    dataset = create_dataset(model_cfg, dataset_cfg, train=False, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)