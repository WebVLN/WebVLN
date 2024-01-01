import argparse
import json

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--gt_caption", type=str, default='Downloads/Data/seen/test.json')
    parser.add_argument("--pd_caption", type=str, default='logs/eval/snap/submit_test.json')
    args = parser.parse_args()

    ptb_tokenizer = PTBTokenizer()

    scorers = [(Cider(), "C"), (Spice(), "S"),
               (Bleu(4), ["B1", "B2", "B3", "B4"]),
               (Meteor(), "M"), (Rouge(), "R")]
    # scorers = [(Cider(), "C")]
# ptb_tokenizer.tokenize({"1":[{'caption':"hi, my name is adam."}]})
#Cider().compute_score({'1': ['hi my name is adam']},{'1': ['hi my name is adam']})
    print(f"loading ground-truths from {args.gt_caption}")
    with open(args.gt_caption) as f:
        gt_captions = json.load(f)
    gt = {}
    for i in gt_captions:
        gt[str(i["idx"])]=[i["QA"][1]]
    # gt_captions = ptb_tokenizer.tokenize(gt_captions)

    print(f"loading predictions from {args.pd_caption}")
    with open(args.pd_caption) as f:
        pred_dict = json.load(f)
    pd = {}
    # for i in pred_dict:
    #     if i["trajectory"][-1] != gt_captions[i["idx"]]["path"][-1]:
    #         gt.pop(str(i["idx"]))
    #         continue
    #     pd[str(i["idx"])]=[i["answer"]]

    for i in pred_dict:
        pd[str(i["idx"])]=[i["answer"]]
        if i["trajectory"][-1] != gt_captions[i["idx"]]["path"][-1]:
            pd[str(i["idx"])]=[""]

    # for level, v in pred_dict.items():
    #     pd_captions[level] = ptb_tokenizer.tokenize(v)

    # print("Start evaluating")
    # score_all_level = list()
    # for level, v in pd_captions.items():
    scores = {}
    for (scorer, method) in scorers:
        score, score_list = scorer.compute_score(gt, pd)
        if type(score) == list:
            for m, s in zip(method, score):
                scores[m] = s
        else:
            scores[method] = score
        # if method == "C":
            # score_all_level.append(np.asarray(score_list))

    # print(
    #     ' '.join([
    #         "C: {C:.4f}", "S: {S:.4f}",
    #         "M: {M:.4f}", "R: {R:.4f}",
    #         "B1: {B1:.4f}", "B2: {B2:.4f}",
    #         "B3: {B3:.4f}", "B4: {B4:.4f}"
    #     ]).format(
    #         C=scores['C'], S=scores['S'],
    #         M=scores['M'], R=scores['R'],
    #         B1=scores['B1'], B2=scores['B2'],
    #         B3=scores['B3'], B4=scores['B4']
    #     ))
    print(
        ' '.join([
            "B1: {B1:.4f}", "B4: {B4:.4f}",
            "C: {C:.4f}", "M: {M:.4f}",
            "R: {R:.4f}","S: {S:.4f}"
            
        ]).format(
            B1=scores['B1']*100, B4=scores['B4']*100,
            C=scores['C']*100,   M=scores['M']*100,
            R=scores['R']*100,   S=scores['S']*100
        ))
    print(
        ''.join([
            "&{B1:.2f}", "&{B4:.2f}",
            "&{C:.2f}",  "&{M:.2f}",
            "&{R:.2f}",  "&{S:.2f}",
            
        ]).format(
            B1=scores['B1']*100, B4=scores['B4']*100,
            C=scores['C']*100,   M=scores['M']*100,
            R=scores['R']*100,   S=scores['S']*100
            
        ))

    # score_all_level = np.stack(score_all_level, axis=1)
    # print(
    #     '  '.join([
    #         "4 level ensemble CIDEr: {C4:.4f}",
    #         "3 level ensemble CIDEr: {C3:.4f}",
    #         "2 level ensemble CIDEr: {C2:.4f}",
    #     ]).format(
    #         C4=score_all_level.max(axis=1).mean(),
    #         C3=score_all_level[:, :3].max(axis=1).mean(),
    #         C2=score_all_level[:, :2].max(axis=1).mean(),
    #     ))