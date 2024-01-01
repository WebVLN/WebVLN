''' Evaluation of agent trajectories '''

import json
from collections import defaultdict

import pprint
pp = pprint.PrettyPrinter(indent=4)

from tqdm import tqdm
import multiprocessing as mp

from calculate_wups import calculate_wups



class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, data_dir, data=None):
        self.scores = defaultdict(list)
        self.splits = splits
        self.gt = {}
        self.ids = []
        # self.scans = []
        for split in splits:
            for item in data:
                # if scans is not None and item['scan'] not in scans:
                #     continue
                self.gt[str(item['idx'])] = item
                # self.scans.append(item['scan'])
                self.ids.append(item['idx'])
        # self.scans = set(self.scans)
        # self.instr_ids = set(instr_ids)
        self.shortest_paths = json.load(open(f"{data_dir}/shortest_paths.json", 'r'))
        # map_dir = f"{data_dir}/map.json"
        # self.map = json.load(open(map_dir, 'r'))

    def _score_item(self, instr_id, path, answer):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        gt = self.gt[str(instr_id)]
        start = gt['path'][0]
        assert start == path[0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        gt_answer = gt['A']
        final_position = path[-1] 
        self.scores['tl'].append(len(path))

        if final_position == goal:
            self.scores['success'].append(1)
            self.scores['spl'].append(len(gt['path']) / max(len(gt['path']), len(path)) )
            # if gt_answer == answer:
            #     self.scores['qa_acc'].append(1)
            # else:
            #     self.scores['qa_acc'].append(0)
            wups_9 = calculate_wups(gt_answer, answer, thresh=0.9)
            wups_0 = calculate_wups(gt_answer, answer, thresh=0.0)
            self.scores['wups_0.9'].append(wups_9)
            self.scores['wups_0.0'].append(wups_0)
        else:
            self.scores['success'].append(0)
            self.scores['spl'].append(0)
            # self.scores['qa_acc'].append(0)
            self.scores['wups_0.9'].append(0)
            self.scores['wups_0.0'].append(0)

        if goal in set(path):
            self.scores['oracle_success'].append(1)
        else:
            self.scores['oracle_success'].append(0)

        # self.scores['shortest_lengths'].append(
        #     self.distances[gt['scan']][start][goal]
        # )

    def score(self, results):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        ids = set(self.ids)

        print('result length', len(results))

        
        # print("Start Calculating Scores")
        # pool = mp.Pool()
        # ps = []
        # for r in results:
        #     if r['idx'] in self.ids_:
        #         self.ids_.remove(r['idx'])
        #         p = pool.apply_async(self._score_item, (r['idx'], r['trajectory'], r['answer']))
        #         ps.append(p)
        # ps = [i.get() for i in tqdm(ps)]
        # pool.close()
        # pool.join()
        # for i in ps:
        #     self.scores['success'].append(i['success'][0])
        #     self.scores['oracle_success'].append(i['oracle_success'][0])
        #     self.scores['tl'].append(i['tl'][0])
        #     self.scores['spl'].append(i['spl'][0])
        #     self.scores['qa_acc'].append(i['qa_acc'][0])
        #     self.scores['wups_0.9'].append(i['wups_0.9'][0])
        #     self.scores['wups_0.0'].append(i['wups_0.0'][0])

        for item in tqdm(results):
            # Check against expected ids
            if item['idx'] in ids:
                ids.remove(item['idx'])
                self._score_item(item['idx'], item['trajectory'], item['answer'])

        # if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
        #     assert len(ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
        #                    % (len(ids), len(self.ids), ",".join(self.splits), output_file)
        #     assert len(self.scores['nav_errors']) == len(self.ids)
        import numpy as np
        score_summary = {
            'SR': np.average(self.scores['success'])*100,
            'OSR': np.average(self.scores['oracle_success'])*100,
            'TL': np.average(self.scores['tl']),
            'SPL': np.average(self.scores['spl'])*100,
            # 'QA_acc': np.average(self.scores['qa_acc'])*100,
            'WUPS0.9': np.average(self.scores['wups_0.9'])*100,
            'WUPS0.0': np.average(self.scores['wups_0.0'])*100,
            # 'lengths': np.average(self.scores['trajectory_lengths'])
        }
        # num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        # score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        # oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        # score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        # spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
        #     for error, p, l in
        #     zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        # ]
        # score_summary['spl'] = np.average(spl)

        return score_summary, self.scores
