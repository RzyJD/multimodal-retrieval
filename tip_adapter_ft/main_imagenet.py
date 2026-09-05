"""Target-versus-background adaptation derived from Tip-Adapter.

The scoring and evaluation protocol are historical. Publication cleanup changes
path resolution, explicit preparation and cache placement, not reported results.
"""

import os
import json
from pathlib import Path
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from imagenet import ImageNet
import clip
from utils import *
from PIL import ImageFile
from preprocessing import process_dataset, MARKER, OWNER
# Historical image-loading setting: allow truncated JPEG files.
ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        default=str(Path(__file__).with_name('imagenet.yaml')),
        help='YAML configuration; relative data paths resolve from the YAML directory'
    )
    parser.add_argument('--prepare-data', action='store_true',
                        help='Explicitly generate the binary dataset before training')
    parser.add_argument('--overwrite-prepared-data', action='store_true',
                        help='Replace generated data; requires --prepare-data')
    args = parser.parse_args()

    if args.overwrite_prepared_data and not args.prepare_data:
        parser.error('--overwrite-prepared-data requires --prepare-data')
    return args
# The configured run evaluates one target class against Others.
#给定类别target，利用掩码返回该target下的正样本和负样本
def mask(logits,true,target,device='cuda'):
    with torch.no_grad():
    #寻找图片对target类的logits
        pred=torch.softmax(logits,dim=1)
        scores=pred[:,target].cpu().numpy()
       # print(pred)
        #print(pred.shape)
        #scores=logits[:,target].cpu().numpy()
        true=true.cpu().numpy()
    #实际为target类的mask
        pos_mask=(true==target)
    #实际不为target类的mask
        neg_mask=(true!=target)
    #提取score中为target类的图片对target的logits
        pos=scores[pos_mask]
    #提取score中不为target类的图片对target的logits
        neg=scores[neg_mask]
        return pos,neg

#计算评估指标
def metrics(pos,neg,threshold):
    #计算TP,FP,FN
    pos=np.array(pos)
    neg=np.array(neg)
    TP=sum(pos>=threshold)
    FP=sum(neg>=threshold)
    FN=sum(pos<threshold)
    #防止除零错误
    epsilon=1e-6
    precision=TP/(TP+FP+epsilon)
    recall=TP/(TP+FN+epsilon)
    f1=2*precision*recall/(precision+recall)
    return f1,precision,recall

#寻找最佳阈值
def find_threshold(pos,neg,verbose=False):
    thresholds=np.linspace(0,1,100)
    best_threshold=0
    best_f1_score=0
    best_precision=0
    best_recall=0
    for threshold in thresholds:
        f1,precision,recall=metrics(pos,neg,threshold)
        if f1 > best_f1_score:
            best_f1_score=f1
            best_threshold=threshold
            best_precision=precision
            best_recall=recall
    if verbose:
        print(f'f1:{best_f1_score},precision:{best_precision},recall:{best_recall}')
    return best_f1_score
    
'''
def find_threshold(pos, neg,verbose=False):
    #确定搜索阈值范围，将范围确定在正样本和负样本的评分区间的并集，确保有效搜索
    min_val = max(min(pos), min(neg))
    max_val = min(max(pos), max(neg))
    num_vals = int((max_val-min_val)*10)
    thresholds = np.linspace(min_val, max_val, num_vals)
    best_threshold = 0.
    best_f1_score = 0.
    best_precision = 0.
    best_recall = 0.
    f1_scores = []

    for threshold in thresholds:
        f1_score, precision, recall = metrics(pos, neg, threshold)
        f1_scores.append(f1_score)

        # 判断最佳f1
        if f1_score > best_f1_score:
            # 更新指标
            best_threshold = threshold
            best_f1_score = f1_score
            best_precision = precision
            best_recall = recall
    if verbose:
        print(f'f1:{best_f1_score},precision:{best_precision},recall:{best_recall}')
    return best_f1_score'''


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights,class_names):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    for i,class_name in enumerate(class_names):
        if class_name != "Others":
            pos,neg=mask(clip_logits,test_labels,i)
            f1=find_threshold(pos,neg,verbose=True)
    print("\n**** Zero-shot CLIP's test F1: {:.4f}. ****\n".format(f1))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    for i, class_name in enumerate(class_names):
        if class_name != "Others":
            pos, neg = mask(tip_logits, test_labels, i)
            f1 = find_threshold(pos, neg)
    print("**** Tip-Adapter's test F1: {:.4f}. ****\n".format(f1))

    # Search Hyperparameters
    #_ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights,class_names)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * best_alpha

    for i,class_name in enumerate(class_names):
        if class_name != "Others":
            pos,neg=mask(tip_logits,test_labels,i)
            f1=find_threshold(pos,neg,verbose=True)
    print("**** Tip-Adapter's test F1: {:.4f}. ****\n".format(f1))
    return best_beta, best_alpha
def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F,class_names,beta,alpha):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    #beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_f1, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            loss = F.cross_entropy(tip_logits, target)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        for i, class_name in enumerate(class_names):
            if class_name != "Others":
                pos, neg = mask(tip_logits, test_labels, i)
                f1 = find_threshold(pos, neg)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"** After fine-tuning, Tip-Adapter-F's best test f1: {best_f1:.2f}, at epoch: {best_epoch}. **\n")

    # Search Hyperparameters
    #_ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter)
    adapter.eval()
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights,class_names,adapter=adapter)
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    clip_logits = 100. * test_features @ clip_weights
    tip_logits = clip_logits + cache_logits * best_alpha
    for i, class_name in enumerate(class_names):
        if class_name != "Others":
            pos_res, neg_res = mask(tip_logits, test_labels, i)
            print(f'{class_name}')
            f1= find_threshold(pos_res, neg_res,verbose=True)
    print("** Tip-Adapter-F's test F1: {:.4f}. **\n".format(f1))

def main():
    # Load config file
    args = get_arguments()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open(encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    for key in ('root_path', 'source_dir', 'target_dir'):
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((config_path.parent / path).resolve())
    expected_target = Path(cfg['root_path']) / 'datasets' / 'images'
    if Path(cfg['target_dir']) != expected_target:
        raise ValueError('target_dir must be root_path/datasets/images; see DATASET.md')
    if cfg['keep_classes'] not in {'Dog', 'Duck', 'Erhu', 'Piano', 'Porcelain'}:
        raise ValueError('keep_classes must be one of the five documented target classes')
    class_names =[cfg['keep_classes'],'Others']
    imagenet_classes=class_names
    if args.prepare_data:
        process_dataset(cfg['source_dir'], cfg['target_dir'], cfg['keep_classes'],
                        overwrite=args.overwrite_prepared_data)
    if not expected_target.is_dir():
        raise FileNotFoundError('Prepared data not found. See DATASET.md and use --prepare-data explicitly.')
    marker_path = expected_target / MARKER
    if not marker_path.is_file():
        raise ValueError('Prepared data metadata is missing; regenerate with preprocessing.py.')
    marker = json.loads(marker_path.read_text())
    if marker.get('owner') != OWNER or marker.get('target_class') != cfg['keep_classes']:
        raise ValueError('Prepared data belong to a different target; regenerate for keep_classes.')
    if not torch.cuda.is_available():
        raise RuntimeError('This historical Tip-Adapter implementation requires CUDA.')
    cache_dir = (Path(cfg['root_path']) / 'caches' / cfg['keep_classes'] /
                 cfg['backbone'].replace('/', '-') / f"{cfg['shots']}shots")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg['cache_dir'] = str(cache_dir)

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess,imagenet_classes)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    prompt_templates = {
        "Dog": {
            "positive": [
                # 基础与场景
                "a photo of a dog",
                "a photo of a pet dog sleeping on a couch",
                "a playful dog running on the grass",
                # 特征描述
                "a dog with fluffy fur and a wagging tail",
                "a close-up of a dog's wet nose and pointy ears",
                "a canine with four legs and a shaggy coat",
                "a photo of a dog showing its paws"
            ],
            "negative": [
                # 基础难负例
                "a photo of a wolf",
                "a photo of a cat",
                "a photo of a fox",
                # 特征描述（强调与狗的区别）
                "a wild canine with a long snout, a wolf",
                "a feline with whiskers and retractable claws, a cat",
                "a wild animal with a large bushy tail, a fox",
                "an animal known for its solitary hunting, a wolf",
                # 强调“非正类”的笼统描述
                "a photo of a thing that is not a dog"
            ]
        },
        "Duck": {
            "positive": [
                # 基础与场景
                "a photo of a duck",
                "a duck swimming peacefully in a pond",
                "a flock of ducks flying in the sky",
                # 特征描述
                "a waterfowl with a broad flat beak and webbed feet",
                "a photo of a duck with colorful, waterproof feathers",
                "a close-up of a duck's orange bill",
            ],
            "negative": [
                # 基础难负例
                "a photo of a goose",
                "a photo of a swan",
                "a photo of a seagull",
                # 特征描述（强调与鸭的区别）
                "a large white bird with a long, elegant, curved neck, a swan",
                "a gray waterfowl with a longer neck and a more aggressive stance than a duck, a goose",
                "a coastal bird with a sharp, pointed beak, a seagull",
                # 强调“非正类”的笼统描述
                "a photo of thing that is not related to a duck"
            ]
        },
        "Erhu": {
            "positive": [
                # 基础与场景
                "a photo of an erhu",
                "a musician playing an erhu with a bow",
                "a traditional Chinese musical instrument, the erhu",
                # 特征描述
                "a two-stringed instrument with a long thin neck and a hexagonal soundbox",
                "a close-up of the erhu's snakeskin resonator and wooden body",
                "a bowed fiddle that produces a melancholy sound",
                "an erhu resting on its stand"
            ],
            "negative": [
                # 基础难负例
                "a photo of a violin",
                "a photo of a cello",
                "a photo of a guitar",
                # 特征描述（强调与二胡的区别）
                "a four-stringed instrument with f-holes, played under the chin, a violin",
                "a large string instrument that rests on the floor, played while seated, a cello",
                "a six-stringed instrument that is typically plucked or strummed, a guitar",
                # 强调“非正类”的笼统描述
                "a photo of a thing that is not an erhu"
            ]
        },
        "Piano": {
            "positive": [
                # 基础与场景
                "a photo of a grand piano",
                "a photo of grand piano on stage"
                # 特征描述
                "a photo of a polished black grand piano with its lid open",
                "a photo of a keyboard instrument where hammers strike strings to produce sound",
            ],
            "negative": [
                # 基础难负例
                "a photo of an organ",
                "a photo of a harpsichord",
                "a photo of an accordion"
                "a photo of an electronic keyboard, lacking a wooden soundboard or hammer mechanism. Classified as non-piano.",
                # 特征描述（强调与钢琴的区别）
                "a photo of a musical instrument with large vertical pipes that produce sound, a pipe organ",
                "a photo of an early keyboard instrument where strings are plucked, not struck, a harpsichord",
                "a photo of a portable instrument with a keyboard and bellows, an accordion",
                # 强调“非正类”的笼统描述
                "an image of an object that is not a piano"
            ]
        },
         "Porcelain": {
        "positive": [
            # 核心工艺与原料（硬特征）
            "a porcelain made from kaolin clay (no t red clay) fired at ≥1200°C (high-fired)",
            "a vitrified ceramic with mullite crystals, formed by kaolin-based paste",
            # 物理特性对比（区分玻璃/陶器）
            "a translucent porcelain (not fully transparent like glass), showing soft light transmission",
            "a hard ceramic (Mohs hardness 7) that can scratch glass",
            # 显式排除混淆项
            "a porcelain, distinct from earthenware (low-fired) and glass (silica-based)",
            # 典型案例强化
            "a blue and white porcelain vase with underglaze painting, fired in Jingdezhen kilns"
        ],
        "negative": [
            # 重点混淆项1：陶器（Earthenware）
            "a low-fired earthenware pot made from red clay (<1000°C), porous and opaque",
            "terracotta sculpture with rough surface, lacking kaolin and high-firing",
            # 重点混淆项2：玻璃器（Glass）
            "a fully transparent glass vase made from silica sand, no clay component",
            "a glass object with smooth surface but no ceramic texture, melting point <800°C",
            # 易分辨负类（粗粒度概括）
            "a non-ceramic object",
            "a photo of living creature (not ceramic)"
        ]

        }
    }

    clip_weights = clip_classifier(class_names, prompt_templates[cfg['keep_classes']], clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    best_beta,best_alpha=run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights,class_names)

    # ------------------------------------------ Tip-Adapter-F -------------------/-----------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F,class_names,best_beta,best_alpha)
           

if __name__ == '__main__':
    main()
