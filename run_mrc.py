import argparse
import os
import logging
from attrdict import AttrDict
import warnings

from src.functions.modules import QUESTION_ANSWERING_MODEL, CONFIG, TOKENIZER
from src.model.main_functions import train, evaluate
from src.functions.utils import init_logger, set_seed
warnings.filterwarnings(action='ignore')


def create_model(args):

    config = CONFIG[args.model_type].from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
    )

    tokenizer = TOKENIZER[args.model_type].from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        do_lower_case=args.do_lower_case, use_fast=False if args.model_type == 'roberta' else True
    )
    model = QUESTION_ANSWERING_MODEL["roberta"].from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        config=config,
    )

    # # vocab 추가
    # if args.add_vocab:
    #     add_token = {"additional_special_tokens": []}
    #     tokenizer.add_special_tokens(add_token)
    #     model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    return model, tokenizer


def main(cli_args):

    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    init_logger()
    set_seed(args)

    model, tokenizer = create_model(args)

    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        evaluate(args, model, tokenizer, logger)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

         ### Pre-trained Model Setting ###
    # ############## KoELECTRA v3 ###############
    # cli_parser.add_argument("--model_type", type=str, default="electra")
    # cli_parser.add_argument("--model_name_or_path", type=str, default="monologg/koelectra-base-v3-discriminator")
    #
    # ############### KLUE RoBERTa base ###############
    cli_parser.add_argument("--model_type", type=str, default="roberta")
    cli_parser.add_argument("--model_name_or_path", type=str, default="klue/roberta-base")    # "klue/roberta-base"


                     ### DATASET ###
    # ############### KorQUAD  v1.0 ###############
    # cli_parser.add_argument("--data_dir", type=str, default="./data")
    # cli_parser.add_argument("--train_file", type=str, default="KorQuAD_v1.0_dev.json")
    # cli_parser.add_argument("--predict_file", type=str, default="KorQuAD_v1.0_dev.json")

    # ############### KLUE MRC v1.0 ###############
    # cli_parser.add_argument("--data_dir", type=str, default="./KLUE_data")
    # cli_parser.add_argument("--train_file", type=str, default="klue-mrc-v1_train.json")
    # cli_parser.add_argument("--predict_file", type=str, default="klue-mrc-v1_dev.json")

    ############### Paper-QA ###############
    cli_parser.add_argument("--data_dir", type=str, default="./data/seed42")
    cli_parser.add_argument("--train_file", type=str, default="train.json")
    cli_parser.add_argument("--predict_file", type=str, default="val.json")     # train/val/test


    cli_parser.add_argument("--output_dir", type=str, default="./model/roberta_base")
    cli_parser.add_argument("--checkpoint", type=str, default="500")

    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=510)    # 512
    cli_parser.add_argument("--doc_stride", type=int, default=128)
    cli_parser.add_argument("--max_query_length", type=int, default=64)
    cli_parser.add_argument("--max_answer_length", type=int, default=30)
    cli_parser.add_argument("--n_best_size", type=int, default=20)

    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=3e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=16)
    cli_parser.add_argument("--eval_batch_size", type=int, default=32)
    cli_parser.add_argument("--num_train_epochs", type=int, default=5)

    cli_parser.add_argument("--save_steps", type=int, default=2000)
    cli_parser.add_argument("--logging_steps", type=int, default=2000)     # 500
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=16)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # For SQuAD v2.0 (Yes/No Question)
    cli_parser.add_argument("--version_2_with_negative", type=bool, default=True)
    cli_parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default=True)
    cli_parser.add_argument("--add_vocab", type=bool, default=False)
    cli_parser.add_argument("--do_train", type=bool, default=True)
    cli_parser.add_argument("--do_eval", type=bool, default=False)

    cli_args = cli_parser.parse_args()

    main(cli_args)