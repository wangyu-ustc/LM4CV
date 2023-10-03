from utils.train_utils import *
# import open_clip
from sklearn.cluster import KMeans

from utils.train_utils import *


def cluster(cfg):

    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    elif cfg['model_type'] == 'open_clip':
        model, _, preprocess = open_clip.create_model_and_transforms(cfg['model_size'], pretrained=cfg['openclip_pretrain'], device='cuda')
        model = model.cuda()
        tokenizer = open_clip.get_tokenizer(cfg['model_size'])
    else:
        raise NotImplementedError

    attributes = get_attributes(cfg)
    attribute_embeddings = []
    batch_size = 32
    for i in range((len(attributes) // batch_size) + 1):
        sub_attributes = attributes[i * batch_size: (i + 1) * batch_size]
        if cfg['model_type'] == 'clip':
            clip_attributes_embeddings = clip.tokenize([get_prefix(cfg) + attr for attr in sub_attributes]).cuda()
        elif cfg['model_type'] == 'open_clip':
            clip_attributes_embeddings = tokenizer([get_prefix(cfg) + attr for attr in sub_attributes]).cuda()

        attribute_embeddings += [embedding.detach().cpu() for embedding in
                                 model.encode_text(clip_attributes_embeddings)]
        
    attribute_embeddings = torch.stack(attribute_embeddings).float()
    attribute_embeddings = attribute_embeddings / attribute_embeddings.norm(dim=-1, keepdim=True)

    print ("num_attributes: ", cfg['num_attributes'])
    if cfg['num_attributes'] == 'full':
        return attributes, attribute_embeddings

    if cfg['cluster_feature_method'] == 'random':
        selected_idxes = np.random.choice(np.arange(len(attribute_embeddings)), size=cfg['num_attributes'], replace=False)


    elif cfg['cluster_feature_method'] == 'similarity':
        if cfg['model_type'] == 'clip':
            model, preprocess = clip.load(cfg['model_size'])
        else:
            raise NotImplementedError
        train_loader, test_loader = get_image_dataloader(cfg['dataset'], preprocess)
        print("Get Embeddings...")
        train_features = get_image_embeddings(cfg, cfg['dataset'], model, train_loader, 'train')
        test_features = get_image_embeddings(cfg, cfg['dataset'], model, test_loader, 'test')
        if cfg['dataset'] == 'imagenet-animal':
            train_labels, test_labels = get_labels("imagenet")
            train_labels, test_labels = np.array(train_labels), np.array(test_labels)

            train_idxes = np.where((train_labels < 398) & (train_labels!=69))
            train_features = train_features[train_idxes]

            test_idxes = np.where((test_labels < 398) & (test_labels!=69))
            test_features = test_features[test_idxes]
        train_labels, test_labels = get_labels(cfg['dataset'])

        print("Initializing Feature Dataset")
        train_feature_dataset = FeatureDataset(train_features, train_labels)
        train_loader = DataLoader(train_feature_dataset, batch_size=cfg['batch_size'], shuffle=False)
        train_scores = extract_concept_scores(train_loader, model, attribute_embeddings)

        train_scores = np.array(train_scores)
        mean_scores = np.mean(train_scores, axis=0)
        assert len(mean_scores) == len(attribute_embeddings)
        selected_idxes = np.argsort(mean_scores)[::-1][:cfg['num_attributes']].astype(int)

    else:
        if cfg['cluster_feature_method'] == 'linear':

            mu = torch.mean(attribute_embeddings, dim=0)
            sigma_inv = torch.tensor(np.linalg.inv(torch.cov(attribute_embeddings.T)))
            configs = {
                'mu': mu,
                'sigma_inv': sigma_inv,
                'mean_distance': np.mean([mahalanobis_distance(embed, mu, sigma_inv) for embed in attribute_embeddings])
            }

            model = get_model(cfg, cfg['linear_model'], input_dim=attribute_embeddings.shape[-1], output_dim=get_output_dim(cfg['dataset']))
            train_loader, test_loader = get_feature_dataloader(cfg)
            if cfg['mahalanobis']:
                best_model, best_acc = train_model(cfg, cfg['linear_epochs'], model, train_loader, test_loader, regularizer='mahalanobis', configs=configs)
            else:
                if cfg.get("cosine", False):
                    best_model, best_acc = train_model(cfg, cfg['linear_epochs'], model, train_loader, test_loader, regularizer='cosine', configs=configs)

                else:
                    best_model, best_acc = train_model(cfg, cfg['linear_epochs'], model, train_loader, test_loader, regularizer=None, configs=configs)

            centers = best_model[0].weight.detach().cpu().numpy()

        elif cfg['cluster_feature_method'] == 'kmeans':
            kmeans = KMeans(n_clusters=cfg['num_attributes'], random_state=0).fit(attribute_embeddings)
            centers = kmeans.cluster_centers_

        elif cfg['cluster_feature_method'] == 'svd':
            u, s, vh = np.linalg.svd(attribute_embeddings.numpy().astype(np.float32), full_matrices=False)

            u = u[:cfg['num_attributes'], :]
            centers = u @ np.diag(s) @ vh
        else:
            raise NotImplementedError

        selected_idxes = []
        for center in centers:
            center = center / torch.tensor(center).norm().numpy()
            distances = np.sum((attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1)
            # sorted_idxes = np.argsort(distances)[::-1]
            sorted_idxes = np.argsort(distances)
            count = 0
            while sorted_idxes[count] in selected_idxes:
                count += 1
            selected_idxes.append(sorted_idxes[count])
        selected_idxes = np.array(selected_idxes)

    if cfg['cluster_feature_method'] == 'linear':
        return best_acc, best_model, [attributes[i] for i in selected_idxes], torch.tensor(attribute_embeddings[selected_idxes])
    else:
        return [attributes[i] for i in selected_idxes], torch.tensor(attribute_embeddings[selected_idxes])



