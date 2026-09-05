from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from main_imagenet import mask,find_threshold
import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
#对于每一类，用多种不同的templates特征化再取平均，输出每类的embedding向量



def clip_classifier(classnames,prompt_templates, clip_model,x=1):
    with torch.no_grad():
        clip_weights = []
        for i, classname in enumerate(classnames):
            # 处理类名中的下划线
            classname = classname.replace('_', ' ')

            # 为“其他”类创建对比性模板
            if i == x:
                #base = [t.format(classname) for t in template]

                #contrastive_templates = [
                    #"a photo of something that is not a {}"]

                contrastive_templates = prompt_templates['negative']
                texts = [t.format(classnames[0]) for t in contrastive_templates]
                print(texts)
            else:
                # 正样本类使用原始模板
                texts = [t.format(classname) for t in prompt_templates['positive']]
                print(texts)

            # 编码文本
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            # 特征归一化和平均
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            clip_weights.append(class_embedding)

        # 转置以方便矩阵乘法
        clip_weights = torch.stack(clip_weights, dim=1).cuda()

    return clip_weights

def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()


        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

#将loader中图片的特征预先提取并储存，避免重复加载
def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                # i每个批次每个批次地迭代
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, class_names,adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.01) / cfg['search_step'][0] + 0.01 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.01) / cfg['search_step'][1] + 0.01 for i in range(cfg['search_step'][1])]

        best_beta, best_alpha = 0, 0
        best_f1=0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                for i, class_name in enumerate(class_names):
                    if class_name != "Others":
                        pos, neg = mask(tip_logits, labels, i)
                        f1 = find_threshold(pos, neg)
                if f1 > best_f1:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; f1: {:.4f}".format(beta, alpha, f1))
                    best_f1 = f1
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best f1: {:.4f}.\n".format(best_f1))
        print("\nAfter searching, the best beta: {:.4f}, the best alpha: {:.4f}.\n".format(best_beta, best_alpha))
    else:
            best_beta = cfg["init_beta"]
            best_alpha = cfg["init_alpha"]

    return best_beta, best_alpha
