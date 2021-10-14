import logging
import random
import torch
import numpy as np
import os
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from src.functions.processor import SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_examples(args, tokenizer, evaluate=False, output_examples=False):
    input_dir = args.data_dir
    print("Creating features from dataset file at {}".format(input_dir))

    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

    if evaluate:
        examples = processor.get_dev_examples(os.path.join(args.data_dir),
                                              filename=args.predict_file)
    else:
        examples = processor.get_train_examples(os.path.join(args.data_dir),
                                                filename=args.train_file)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )
    if output_examples:
        return dataset, examples, features
    return dataset