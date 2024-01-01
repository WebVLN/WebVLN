import torch

import os
import time
import json
from collections import defaultdict
import pickle as pkl

from utils import timeSince, print_progress
from utils import setup_seed, read_img_features, prepare_dataset

from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

log_dir = 'logs/%s/snap/' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


print(args); print('')


def build_dataset(args):
    # Load the env img features
    img_feat_dict = read_img_features(args.data_dir+"/img_feats.pkl", test_only=args.test_only)
    text_feat_dict = pkl.load(open(f"{args.data_dir}/text_feats.pkl", 'rb')) 
    screenshot_feat_dict = pkl.load(open(f"{args.data_dir}/screenshot_crop_feats.pkl", 'rb')) 
    # Create the training environment
    train_episodes = prepare_dataset(args, splits=['train'])
    train_env = R2RBatch(
        img_feat_dict, 
        text_features=text_feat_dict,
        screenshot_features=screenshot_feat_dict,
        data_dir=args.data_dir,
        batch_size=args.batchSize,
        splits=['train'],
        data=train_episodes, )

    # Setup the validation data
    val_env_names = ["val","test"]
    val_envs = {}
    for split in val_env_names:
        val_episodes = prepare_dataset(args, splits=[split])
        val_envs[split] = (
            R2RBatch(
                img_feat_dict, 
                text_features=text_feat_dict,
                screenshot_features=screenshot_feat_dict,
                data_dir=args.data_dir,
                batch_size=args.batchSize,
                splits=[split],
                data=val_episodes,
            ),
            Evaluation([split], data_dir=args.data_dir, data=val_episodes),
        )

    return train_env, val_envs


''' train the listener '''
def train(train_env, val_envs, n_iters=1000, log_every=10):
    writer = SummaryWriter(log_dir=log_dir)
    record_file = open('./logs/%s/'%(args.name) + 'train_logs.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    listner = Seq2SeqAgent(
        train_env, results_path="", episode_len=args.maxAction)

    start_iter = 0
    if args.load is not None:
        start_iter = listner.load(os.path.join(args.load))
        print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val': {"SR": 0., "SPL": 0.,"WUPS0.9": 0.,"WUPS0.0": 0., "state":""}}

    # args.eval_first=True
    # if args.eval_first:
    #     idx = 0
    #     loss_str = "validation before training"
    #     for env_name, (env, evaluator) in val_envs.items():
    #         listner.env = env
    #         listner.test(use_dropout=False, feedback='argmax', iters=None)
    #         results = listner.get_results()
    #         IL_loss = sum(listner.logs['val_IL_loss']) / max(len(listner.logs['val_IL_loss']), 1)
    #         writer.add_scalar(f"loss/{env_name}_IL_loss", IL_loss, idx)
    #         score_summary, _ = evaluator.score(result)
    #         loss_str += ", %s " % env_name
    #         for metric, val in score_summary.items():
    #             loss_str += ', %s: %.2f' % (metric, val)
    #             writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
    #         record_file = open('./logs/%s/eval.txt'%(args.name), 'a')
    #         record_file.write(loss_str + '\n')
    #         record_file.close()
    #     print(loss_str)


    for idx in range(start_iter, start_iter+n_iters, log_every):
        interval = min(log_every, n_iters-idx)
        iter = idx + interval
        listner.logs = defaultdict(list)
        listner.env = train_env
        listner.train(interval, feedback=args.feedback, writer=writer, idx=idx)  # Train interval iters
        if idx+log_every>=100000:
            for param_group in listner.vln_bert_optimizer.param_groups:
                param_group["lr"] *= 0.9
            for param_group in listner.qa_optimizer.param_groups:
                param_group["lr"] *= 0.9
        # Run validation
        if idx+log_every>=140000:
            loss_str = "iter {}".format(iter)
            for env_name, (env, evaluator) in val_envs.items():
                listner.env = env

                # Get validation distance from goal under test evaluation conditions
                listner.test(use_dropout=False, feedback='argmax', iters=args.eval_iters)
                result = listner.get_results()

                IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
                writer.add_scalar(f"loss/{env_name}_IL_loss", IL_loss, iter)
                score_summary, _ = evaluator.score(result)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.4f' % (metric, val)
                    writer.add_scalar(f'metrics_{env_name}/{metric}', score_summary[metric], iter)

                    # select model by spl+sr
                    if env_name in best_val:
                        if score_summary['WUPS0.9'] + score_summary['SR'] >= best_val[env_name]['WUPS0.9'] + best_val[env_name]['SR']:
                            best_val[env_name]['SR'] = score_summary['SR']
                            best_val[env_name]['SPL'] = score_summary['SPL']
                            best_val[env_name]['WUPS0.9'] = score_summary['WUPS0.9']
                            best_val[env_name]['WUPS0.0'] = score_summary['WUPS0.0']
                            best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                            listner.save(idx, os.path.join('./logs/%s'%(args.name), "state_dict", "best_%s" % (env_name)))

                    record_file = open('./logs/%s/eval.txt'%(args.name), 'a')
                    record_file.write(loss_str + '\n')
                    record_file.close()

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                iter, float(iter)/n_iters*100, loss_str)))
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])
                record_file = open('./logs/%s/eval.txt'%(args.name), 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()


def valid(train_env, val_envs):

    agent = Seq2SeqAgent(
        train_env, results_path="", episode_len=args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (
        agent.load(args.load), args.load))
    loss_str = "iter {}".format(iter)
    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = -1
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)
            record_file = open('./logs/%s/eval_test.txt'%(args.name), 'a')
            record_file.write(loss_str + '\n')
            record_file.close()
            
        json.dump(
            result,
            open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
            sort_keys=True, indent=4, separators=(',', ': ')
        )


if __name__ == "__main__":

    setup_seed()

    train_env, val_envs = build_dataset(args)

    if args.train == 'train':
        train(train_env, val_envs, n_iters=args.iters, log_every=args.log_every)
    elif args.train == 'valid':
        valid(train_env, val_envs)
    else:
        assert False
