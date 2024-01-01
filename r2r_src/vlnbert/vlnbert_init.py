# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au
import torch
from pytorch_transformers import (BertConfig, BertTokenizer)

def get_tokenizer(args):
    if args.vlnbert == 'oscar':
        tokenizer_class = BertTokenizer
        model_name_or_path = 'Oscar/pretrained_models/base-no-labels/ep_67_588997'
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    elif args.vlnbert == 'prevalent':
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    return tokenizer

def get_vlnbert_models(args):
    config_class = BertConfig

    if args.vlnbert == 'oscar':
        from vlnbert.vlnbert_OSCAR import VLNBert
        model_class = VLNBert
        model_name_or_path = 'Downloads/Oscar/pretrained_models/base-no-labels/ep_67_588997'
        vis_config = config_class.from_pretrained(model_name_or_path, num_labels=2, finetuning_task='vln-r2r')

        vis_config.model_type = 'visual'
        vis_config.finetuning_task = 'vln-r2r'
        vis_config.hidden_dropout_prob = 0.3
        vis_config.hidden_size = 768
        vis_config.img_feature_dim = args.feature_size
        vis_config.num_attention_heads = 12
        vis_config.num_hidden_layers = 12
        visual_model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=vis_config)

    elif args.vlnbert == 'prevalent':
        from vlnbert.vlnbert_PREVALENT import VLNBert
        model_class = VLNBert
        # model_name_or_path = 'Prevalent/pretrained_model/pytorch_model.bin'
        model_name_or_path = 'Downloads/Prevalent/model_LXRT.pth'
        vis_config = config_class.from_pretrained('bert-base-uncased')
        vis_config.img_feature_dim = args.feature_size
        vis_config.img_feature_type = ""
        vis_config.vl_layers = 2
        vis_config.la_layers = 9

        # visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config)
        visual_model = model_class(config=vis_config)
        
        # if 'model_LXRT' in model_name_or_path:
        #     checkpoint = {}
        #     tmp = torch.load(model_name_or_path, map_location=lambda storage, loc: storage)
        #     for param_name, param in tmp.items():
        #         param_name = param_name.replace('module.bert.', '')
        #         if 'encoder.layer' in param_name:
        #             param_name = param_name.replace('encoder.layer', 'lalayer')
        #             checkpoint[param_name] = param
        #         elif 'encoder.x_layers' in param_name:
        #             param_name = param_name.replace('encoder.x_layers', 'addlayer')
        #             checkpoint[param_name] = param
        #             checkpoint[param_name] = param
        #         else:
        #             checkpoint[param_name] = param
        #     del tmp
        # visual_model.load_state_dict(checkpoint, strict=False)

    return visual_model
