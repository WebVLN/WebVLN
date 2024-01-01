#!/usr/bin/env python

"""
Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de

The script assumes there are two files
- first file with ground truth answers
- second file with predicted answers
both answers are line-aligned

The script also assumes that answer items are comma separated.
For instance, chair,table,window

It is also a set measure, so not exactly the same as accuracy 
even if dirac measure is used since {book,book}=={book}, also {book,chair}={chair,book}

Logs:
    05.09.2015 - white spaces surrounding words are stripped away so that {book, chair}={book,chair}
"""

import nltk
nltk.download('wordnet')
from numpy import prod
from nltk.corpus import wordnet as wn
from putils import *

def file2list(filepath):
    with open(filepath,'r') as f:
        lines =[k for k in 
            [k.strip() for k in f.readlines()] 
        if len(k) > 0]
    return lines


def list2file(filepath,mylist):
    mylist='\n'.join(mylist)
    with open(filepath,'w') as f:
        f.writelines(mylist)


def items2list(x):
    """
    x - string of comma-separated answer items
    """
    return [l.strip() for l in x.split(',')]


def fuzzy_set_membership_measure(x,A,m):
    """
    Set membership measure.
    x: element
    A: set of elements
    m: point-wise element-to-element measure m(a,b) ~ similarity(a,b)

    This function implments a fuzzy set membership measure:
        m(x \in A) = max_{a \in A} m(x,a)}
    """
    return 0 if A==[] else max(map(lambda a: m(x,a), A))


def score_it(A,T,m):
    """
    A: list of A items 
    T: list of T items
    m: set membership measure
        m(a \in A) gives a membership quality of a into A 

    This function implements a fuzzy accuracy score:
        score(A,T) = min{prod_{a \in A} m(a \in T), prod_{t \in T} m(a \in A)}
        where A and T are set representations of the answers
        and m is a measure
    """
    if A==[] and T==[]:
        return 1
    score_left=0 if A==[] else prod(list(map(lambda a: m(a,T), A)))
    score_right=0 if T==[] else prod(list(map(lambda t: m(t,A),T)))
    return min(score_left, score_right)


# implementations of different measure functions
def dirac_measure(a,b):
    """
    Returns 1 iff a=b and 0 otherwise.
    """
    if a==[] or b==[]:
        return 0.0
    return float(a==b)


def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score 
###

def calculate_wups(input_gt, input_pred, thresh=0.9):
    # method_name = 'CoR3WinG2COCO'
    # cf = importlib.import_module('config.%s' % method_name)
    # gt_filepath = os.path.join(cf.log_dir, "cocoqa_input_answers.txt")
    # pred_filepath = os.path.join(cf.log_dir, "cocoqa_pred_answers.txt")
    # input_gt = file2list(gt_filepath)
    # input_pred = file2list(pred_filepath)

    if thresh == -1:
        our_element_membership = dirac_measure
    else:
        our_element_membership = lambda x, y: wup_measure(x, y, thresh)

    our_set_membership = \
        lambda x, A: fuzzy_set_membership_measure(x, A, our_element_membership)

    # if thresh == -1:
    #     print('standard Accuracy is used')
    # else:
    #     print('soft WUPS at %1.2f is used' % thresh)
    score_list = [score_it(items2list(ta), items2list(pa), our_set_membership)
                  for (ta, pa) in zip(input_gt, input_pred)]
    final_score = float(sum(score_list)) / float(len(score_list))
    # print('final score of WUPS_%1.2f is %2.2f%%' % (thresh, final_score * 100.0))
    return final_score





gt_answer = "The Mini Modine Legging is currently on sale for $16.10."
answer = "The price of the Mini Modine Legging is 16.1"


wups_9 = calculate_wups(gt_answer, answer, thresh=0.9)
wups_0 = calculate_wups(gt_answer, answer, thresh=0.0)
print(f"9: {round(wups_9*100,2)}")
print(f"0: {round(wups_0*100,2)}")