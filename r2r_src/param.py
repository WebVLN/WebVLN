import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')

        self.parser.add_argument('--iters', type=int, default=300000, help='training iterations')
        self.parser.add_argument('--log_every', type=int, default=1000, help='log frequency')
        self.parser.add_argument('--eval_iters', type=int, default=-1, help='name')
        self.parser.add_argument('--name', type=str, default='default', help='experiment name')
        self.parser.add_argument('--vlnbert', type=str, default='oscar', help='oscar or prevalent')
        self.parser.add_argument('--train', type=str, default='listener')
        self.parser.add_argument('--description', type=str, default='no description\n')
        self.parser.add_argument('--data_dir', type=str, default='_web_vln_data/new_data_gen', help='data_dir')
        self.parser.add_argument('--setting', type=str, default=' ', help='setting')
        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=50, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=512)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--features", type=str, default='evaclip')
        self.parser.add_argument("--qa_w", dest='qa_w', type=float, default=1.)
        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='adamW')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.01)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args

args.description = args.name
args.log_dir = 'logs/%s/snap/' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join(args.log_dir, "debug.log"), 'w')
