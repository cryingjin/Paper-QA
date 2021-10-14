import os
import torch
import timeit
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
# from transformers.data.processors.squad import SquadResult
# from src.functions.squad_metrics import compute_predictions_logits
from src.functions.processor import SquadResult
from src.functions.utils import load_examples, set_seed, to_list
from src.functions.evaluate import squad_evaluate


def train(args, model, tokenizer, logger):

    train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss = 0.0
    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    set_seed(args)

    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type == 'roberta':
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            # loss, strong_loss, weak_loss = outputs[:3]
            loss = outputs[0]

            # gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (global_step + 1) % 500 == 0:
                print(
                    "{} step processed.. Current Total Loss : {}\n".format(
                        (global_step + 1), loss.item(),
                    ))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    results = evaluate(args, model, tokenizer, logger, global_step=global_step)

        mb.write("Epoch {} done".format(epoch + 1))
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, logger, global_step = ""):

    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Evaluated File = {}".format(args.predict_file))

    all_results = []

    start_time = timeit.default_timer()

    for batch in progress_bar(eval_dataloader):

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            if args.model_type == "roberta":
                del inputs["token_type_ids"]

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):

            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            # outputs = [start_logits, end_logits]
            output = [to_list(output[i]) for output in outputs]

            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            start_logits, end_logits = output

            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # output directory 설정
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_dir = args.output_dir

    output_prediction_file = os.path.join(args.output_dir, "qa_predictions_{}.json".format(global_step if global_step is not "" else str("dev")))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step if global_step is not "" else str("dev")))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step if global_step is not "" else str("dev")))
        logger.info("  Evaluation - with version2 with negative -")
    else:
        output_null_log_odds_file = None
        logger.info("  Evaluation - without version2 -")

    # 각 result 값 저장
    torch.save(all_results, os.path.join(output_dir,"all_results"))

    # 결과 예측
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    final_results = squad_evaluate(examples, predictions)

    for key in sorted(final_results.keys()):
        logger.info("  %s = %s", key, str(final_results[key]))

    return final_results