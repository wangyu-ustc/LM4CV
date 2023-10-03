import os
import sys
import pdb
import yaml
from utils.train_utils import *
from cluster import cluster


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='cub_bn.yaml',
                        help='configurations for training')
    parser.add_argument("--outdir", default='./outputs',
                        help='where to put all the results')
    return parser.parse_args()


def main(cfg):

    set_seed(cfg['seed'])
    print(cfg)

    if cfg['cluster_feature_method'] == 'linear' and cfg['num_attributes'] != 'full':
        acc, model, attributes, attributes_embeddings = cluster(cfg)
    else:
        attributes, attributes_embeddings = cluster(cfg)

    if cfg['reinit']  and cfg['num_attributes'] != 'full':
        assert cfg['cluster_feature_method'] == 'linear'
        feature_train_loader, feature_test_loader = get_feature_dataloader(cfg)
        model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(dim=-1, keepdim=True)
        for param in model[0].parameters():
            param.requires_grad = False
        best_model, best_acc = train_model(cfg, cfg['epochs'], model, feature_train_loader, feature_test_loader)

    else:
        model = get_model(cfg, cfg['score_model'], input_dim=len(attributes), output_dim=get_output_dim(cfg['dataset']))
        score_train_loader, score_test_loader = get_score_dataloader(cfg, attributes_embeddings)
        best_model, best_acc = train_model(cfg, cfg['epochs'], model, score_train_loader, score_test_loader)

    return best_model, best_acc


if __name__ == '__main__':

    args = parse_config()

    with open(f"{args.config}", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(cfg)


