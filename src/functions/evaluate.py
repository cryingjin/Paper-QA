'''
evaluate_v1 코드 전반적으로 squad 2.0 버전으로 수정
https://github.com/white127/SQUAD-2.0-bidaf/blob/master/evaluate-v2.0.py 참고
rouge score 추가 -> 완료
find_best_thresh_v2 / find_all_best_thresh_v2 -> 삭제
'''

import collections
import logging
from sklearn.metrics import accuracy_score

from src.functions.squad_metrics import compute_exact, compute_f1, compute_rouge_w, normalize_answer

logger = logging.getLogger(__name__)

def get_raw_scores(examples, preds):

    exact_scores = {}
    f1_scores = {}
    rouge_w_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print(f"Missing prediction for {qas_id}")
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
        rouge_w_scores[qas_id] = max(compute_rouge_w(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores, rouge_w_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, rouge_w_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("rouge-w", 100.0 * sum(rouge_w_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("rouge-w", 100.0 * sum(rouge_w_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, rouge_w_raw, na_probs, qid_to_has_ans):

    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    best_rouge_w, best_rouge_w_thresh = find_best_thresh(preds, rouge_w_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["best_rouge_w"] = best_rouge_w
    main_eval["best_rouge_w_thresh"] = best_rouge_w_thresh


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1, rouge_w = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(exact, no_answer_probs, qas_id_to_has_answer,
                                             no_answer_probability_threshold)
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer,
                                          no_answer_probability_threshold)
    rouge_w_threshold = apply_no_ans_threshold(rouge_w, no_answer_probs, qas_id_to_has_answer,
                                               no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold, rouge_w_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, rouge_w_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, rouge_w_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, rouge_w, no_answer_probs, qas_id_to_has_answer)

    return evaluation