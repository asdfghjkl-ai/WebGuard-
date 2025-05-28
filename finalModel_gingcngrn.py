import time
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from transformers import ConvBertModel, ConvBertTokenizer,ConvBertConfig
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from MultiATT import TAMM
from fuse_module import BiAttentionBlock
import torch.nn as nn
import torchmetrics as tm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, confusion_matrix, roc_curve
import csv
import os
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset
import torch.optim as optim
import random
import pandas as pd
import torch
from GraphGuard.models.graphcnn import GraphCNN
import math
from sklearn.model_selection import StratifiedKFold
from bs4 import BeautifulSoup
import networkx as nx
import numpy as np
from mydivision import hash_func_map, features_func_map, division_func_map, get_node_id
from GraphGuard.models.graphcnn import GraphCNN
from GraphGuard.util import get_time, S2VGraph
from tqdm import tqdm

import os
torch.cuda.empty_cache()  # 释放未使用的缓存

# 数据集路径
graphtype = 'ER'
device = 1  # 指定GPU设备编号

# GNNCERT模型参数
num_layers = 5  # 模型的层数
hidden_dim = 768

# 训练参数
# learning_rate = 0.001 # 学习率
learning_rate = 2e-5 # 学习率
weight_decay = 5e-4
num_epochs = 10 # 训练的总轮数
iters_per_epoch = 5
batch_size = 4 # 32
seed = 42 # 随机种子

# 交叉验证与模型结构
fold_n = 2 # n折交叉验证的折数
fold_idx = 0 # 当前使用二等交叉验证折索引
num_mlp_layers = 2 # 每个MLP的层数
final_dropout = 0.5
# 图池化与聚合参数
graph_pooling_type = "sum" # 图级池化方式
neighbor_pooling_type = "sum"
learn_eps = False

# 对抗训练与图划分
num_group = 5 # 将图划分的子图组数（用于对抗训练）
hash_method = "md5"
features_method = "id"
division_method = "node"
degree_as_tag = 0
features_func = features_func_map.get(features_method, get_node_id)
hash_func = hash_func_map.get(hash_method, hash)
division_func = division_func_map.get(division_method, None)
degree_as_tag = bool(degree_as_tag)
# 图数据参数
num_classes = 2  # 标签的类别数（根据你的数据调整） 良性和恶意
ym_select_sub_num = 4 # 随机抽几个图
# 导入所需模块
from gensim.test.utils import get_tmpfile, common_texts  # 内置测试数据和模型路径
from gensim.models import Word2Vec, KeyedVectors

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)
class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(Linear(num_node_features, hidden_channels), torch.nn.ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), Linear(hidden_channels, num_classes))
        self.conv2 = GINConv(nn2)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)
class GRN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GRN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 图级池化
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
def gnn_train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
def gnn_evaluate(model, loader, device, epoch, csv_file='evaluation_metrics.csv'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # 获取正类概率
            pred = output.argmax(dim=1).cpu().numpy()
            label = data.y.cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label)
            all_probs.extend(probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else np.nan
    pr_auc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else np.nan
    mcc = matthews_corrcoef(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(np.unique(all_labels)) > 1 else np.nan
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    def tpr_at_fpr(fpr, tpr, target_fpr):
        idx = np.searchsorted(fpr, target_fpr)
        if idx >= len(tpr):
            return 0.0
        elif fpr[idx] == target_fpr:
            return tpr[idx]
        elif idx > 0:
            fraction = (target_fpr - fpr[idx - 1]) / (fpr[idx] - fpr[idx - 1])
            return tpr[idx - 1] + fraction * (tpr[idx] - tpr[idx - 1])
        else:
            return tpr[0]
    tpr_at_fpr_0001 = tpr_at_fpr(fpr, tpr, 0.0001)
    tpr_at_fpr_001 = tpr_at_fpr(fpr, tpr, 0.001)
    tpr_at_fpr_01 = tpr_at_fpr(fpr, tpr, 0.01)
    tpr_at_fpr_1 = tpr_at_fpr(fpr, tpr, 0.1)
    metrics = {
        'Epoch': epoch,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'ACC': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'MCC': mcc,
        'Weighted F1': weighted_f1,
        'TPR@FPR=0.0001': tpr_at_fpr_0001,
        'TPR@FPR=0.001': tpr_at_fpr_001,
        'TPR@FPR=0.01': tpr_at_fpr_01,
        'TPR@FPR=0.1': tpr_at_fpr_1,
    }
    df = pd.DataFrame([metrics])
    file_exists = os.path.isfile(csv_file)
    df.to_csv(csv_file, mode='a', header=not file_exists, index=False, encoding='utf-8')
    print(f"Epoch: {epoch:03d}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}, ACC: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, MCC: {mcc:.4f}, Weighted F1: {weighted_f1:.4f}, TPR@FPR=0.0001: {tpr_at_fpr_0001:.4f}, TPR@FPR=0.001: {tpr_at_fpr_001:.4f}, TPR@FPR=0.01: {tpr_at_fpr_01:.4f}, TPR@FPR=0.1: {tpr_at_fpr_1:.4f}")
    return metrics

# 解析HTML并返回DOM树
def parse_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup
def build_dom_graph(soup):
    G = nx.DiGraph()
    stack = [(soup, 0, None)]
    node_id = 1
    node_attributes = {}
    tags = {}
    node_tags = []
    while stack:
        node, current_id, parent_id = stack.pop()
        if node is not None and node.name:
            node_attributes[current_id] = {
                'tag': node.name,  # 标签类型
                'attributes': dict(node.attrs),
                'text': node.get_text(strip=True)
            }
            if not node.name in tags:
                mapped = len(tags)
                tags[node.name] = mapped
            node_tags.append(tags[node.name])
            G.add_node(current_id,**node_attributes[current_id])  # 添加节点 和属性
            if parent_id is not None:
                G.add_edge(parent_id, current_id)
            children = [child for child in node.children if hasattr(child, 'name')]
            for child in reversed(children):
                if child is not None and child.name:
                    stack.append((child, node_id, current_id))
                    node_id += 1
    return G, node_attributes, node_tags

def get_graph_from_html(html_content,label): # html_content就是html的字符串
    soup = parse_html(html_content)
    G, node_attributes,node_tags = build_dom_graph(soup)
    edges_to_remove = [(u, v) for u, v in G.edges() if random.random() < 0.1]

    G.remove_edges_from(edges_to_remove)
    for node_id, node_data in node_attributes.items():
        for key, value in node_data.items():
            node_data[key] = str(value)

    
    model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    wv = model.wv
    def get_text_vector(text):
        words = text.lower().split()
        if hasattr(wv, 'key_to_index'):
            valid_words = [word for word in words if word in wv.key_to_index]
        else:
            valid_words = [word for word in words if word in wv.vocab]
        if not valid_words:
            return np.zeros(wv.vector_size)
        return np.mean(wv[valid_words], axis=0)

    def get_node_feature(node_data):
        attr1_vector = get_text_vector(node_data["tag"])
        attr1_vector = np.concatenate(
            (
                np.array([len(node_data['attributes'])]),  # 转换为一维数组
                np.array([len(node_data['text'])]),
                attr1_vector
            )
        )
        attr2_vector = get_text_vector(node_data["attributes"])
        attr3_vector = get_text_vector(node_data["text"])
        # 按行拼接
        node_feature_vector = np.concatenate((attr1_vector, attr2_vector, attr3_vector), axis=0)
        return node_feature_vector

    node_features = {}
    for node, attributes in node_attributes.items():
         node_features[node] = get_node_feature(attributes)
    feature_matrix = np.stack(list(node_features.values()))
    node_ids = sorted(G.nodes())
    graph = S2VGraph(
        g=G,
        label=label
    )
    graph.node_tags=node_tags
    graph.node_features=torch.FloatTensor(feature_matrix)
    edges = []
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    for u, v in G.edges():
        edges.append([node_index[u], node_index[v]])
        edges.append([node_index[v], node_index[u]])  # 添加反向边
    graph.edge_mat = torch.LongTensor(edges).t().contiguous()
    graph.neighbors = [
        [node_index[neighbor] for neighbor in G.neighbors(node_id)]
        for node_id in node_ids
    ]
    graph.max_neighbor = max(len(n) for n in graph.neighbors) if graph.neighbors else 0
    return graph

from pytorch_pretrained_bert import BertTokenizer
def dataprocess(filename, input_ids, input_types, input_masks, label):

    pad_size = 200

    bert_path = "character_bert_wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    data = pd.read_csv(filename, encoding='utf-8')
    for i, row in tqdm(data.iterrows(), total=len(data)):
        x1 = row['URL']  # Replace with the column name in your CSV file where the text data is located
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # Get input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

        # Pad if short, truncate if long
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # Set segment to 1 for the masked part
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        # 检测 HTML_Content 是否为 NaN
        html = row['HTML_Content']  # 列名为 'HTML_Content'
        if isinstance(html, float) and math.isnan(html):
            print(f"Skipping index {index} due to NaN in HTML_Content")
            continue  # 跳过当前循环
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)

        assert len(ids) == len(masks) == len(types) == pad_size

        y = row['label']
        if y == 1:
            label.append([1])
        else: label.append([0])
def separate_data(graph_list, input_ids, input_types, input_masks, labels, gnndataset ,seed, fold_idx, n=3):

    assert 0 <= fold_idx and fold_idx < n, f"fold_idx must be from 0 to {n-1}."
    assert len(graph_list) == len(input_ids) == len(input_types) == len(input_masks) == len(labels), "All input lists must have the same length."

    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)

    idx_list = []
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        idx_list.append((train_index, test_index))

    train_idx, test_idx = idx_list[fold_idx]

    # 划分图数据集
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    # 划分文本数据
    train_input_ids = [input_ids[i] for i in train_idx]
    test_input_ids = [input_ids[i] for i in test_idx]

    train_input_types = [input_types[i] for i in train_idx]
    test_input_types = [input_types[i] for i in test_idx]

    train_input_masks = [input_masks[i] for i in train_idx]
    test_input_masks = [input_masks[i] for i in test_idx]

    # 划分标签
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    # 划分标签
    train_gnndataset = [gnndataset[i] for i in train_idx]
    test_gnndataset = [gnndataset[i] for i in test_idx]

    return train_graph_list, test_graph_list, train_input_ids, test_input_ids, train_input_types, test_input_types, train_input_masks, test_input_masks, train_labels, test_labels,train_gnndataset,test_gnndataset

class GraphTextDataset(Dataset):
    def __init__(self, graph, label, input_ids, attention_mask, input_types):
        self.graph = graph
        self.label = label
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.input_types = input_types

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        graph = self.graph[idx]
        label = self.label[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        input_type = self.input_types[idx]

        return graph, label, input_id, attention_mask, input_type
# 3. 应用全连接层
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
class myConvBertModel(nn.Module):
    def __init__(self):
        super(myConvBertModel, self).__init__()
        # 加载预训练的 ConvBERT 模型
        config = ConvBertConfig.from_pretrained("convbert-base", output_hidden_states=True)

        # 加载ConvBERT模型
        self.convbert = ConvBertModel.from_pretrained('convbert-base', config=config)

        # 打开参数
        for param in self.convbert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)  # Add a dropout layer
        self.fc = nn.Linear(768, 2)
        self.hidden_size = 768
        self.fuse = nn.Conv1d(2 * self.hidden_size, self.hidden_size, kernel_size=1)

    def forward(self, input_ids, attention_mask, token_type_ids, extract_features=False):

        outputs = self.convbert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)

        hidden_layers = hidden_states[:-1]
        hidden_layers_tensor = torch.stack(hidden_layers, dim=0).permute(1, 0, 2, 3)

        # 借助TAMM多尺度提取特征
        model_tamm = TAMM(channel=12).to(device)
        pos_pooled = model_tamm.forward(hidden_layers_tensor)

        compressed_feature_tensor = torch.mean(pos_pooled, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)
        text_embeddings = compressed_feature_tensor  # torch.Size([1, 768])

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)
        if extract_features:
            return text_embeddings
        else: return out
class FinalModel(nn.Module):
    def __init__(self, gnn_model, bert_model, gnn_hidden_dim, bert_hidden_dim,num_classes):
        super(FinalModel, self).__init__()
        self.gnn_model = gnn_model
        self.bert_model = bert_model

        self.gnn_hidden_dim = gnn_hidden_dim
        self.bert_hidden_dim = bert_hidden_dim

        self.bi_attention = BiAttentionBlock(
            v_dim=gnn_hidden_dim,
            l_dim=bert_hidden_dim,
            embed_dim=768, #注意力嵌入维度，指中间过程注意力机制映射的维度空间
            num_heads=4,
            dropout=0.1
        )
        self.graph_fc = nn.Linear(in_features=302, out_features=768)
        self.fused_fc = nn.Linear(5,1)
        # 最终分类层
        self.classifier = nn.Linear(gnn_hidden_dim + bert_hidden_dim, num_classes)

    def forward(self, batch_graph, input_ids, input_types, input_masks):

        graph_embeddings = self.gnn_model(batch_graph, extract_features=True)

        graph_embeddings[0] = self.graph_fc(graph_embeddings[0])

        graph_embeddings_tensor = torch.stack(graph_embeddings, dim=0).to(device)

        text_embeddings = self.bert_model(input_ids=input_ids, attention_mask=input_masks,token_type_ids=input_types,
                                          extract_features=True)

        text_embeddings = text_embeddings.repeat_interleave(ym_select_sub_num, dim=0) # 3,dim = 0
        text_embeddings = text_embeddings.unsqueeze(0).repeat(5, 1, 1)

        split_tensor1 = torch.split(graph_embeddings_tensor, ym_select_sub_num, dim=1)
        split_tensor2 = torch.split(text_embeddings, ym_select_sub_num, dim=1)
        assert len(split_tensor1) == len(split_tensor2), "拆分后的子张量数量不一致"
        fused_outputs = []
        for sub1, sub2 in zip(split_tensor1, split_tensor2):
            fused_sub1, fused_sub2 = self.bi_attention(sub1, sub2, attention_mask_v=None, attention_mask_l=None)
            fused_output = torch.cat([fused_sub1, fused_sub2], dim=2).to(device)
            fused_output = fused_output.transpose(1, 2).to(device)
            pool = nn.AdaptiveAvgPool1d(1).to(device)  # 平均池化
            fused_output = pool(fused_output).to(device)
            fused_output = fused_output.transpose(1, 2).to(device)
            fused_output = fused_output.squeeze(1).to(device)
            fused_output = fused_output.transpose(0, 1).to(device)
            fused_output = self.fused_fc(fused_output).to(device)
            fused_output = fused_output.transpose(0, 1).to(device)  # (1,1536)
            fused_outputs.append(fused_output)
        fused_outputs = torch.cat(fused_outputs, dim=0)

        # 分类
        logits = self.classifier(fused_outputs) # [batchsize,1536]
        return logits

def check_batch_graphs_nodes(batch_graphs):
    pos = 3
    batch_err = 0
    while pos:
        if batch_graphs[pos-1].g.number_of_nodes() == 0 : batch_err = batch_err + 1
        pos = pos - 1
    if batch_err == 3:
        return True
    else :return False

def train_final_model(model, train_graphs, train_loader, criterion, optimizer, device):
    model.train()
    for idx, (train_input_ids, train_input_masks, train_input_types,train_labels) in enumerate(train_loader):
        train_input_ids, train_input_masks, train_input_types,train_labels = train_input_ids.to(device), train_input_masks.to(device), train_input_types.to(device), train_labels.to(device)

        start_idx = idx * batch_size
        end_idx = (idx + 1) * batch_size - 1
        if end_idx >= true_graph_num: end_idx = true_graph_num - 1  # 如果索引超过
        graphs = train_graphs[start_idx:end_idx + 1]
        graphs_sub = []
        for i, graph in enumerate(graphs):
            subgraphs = sum([division_func(graph, features_func, hash_func, num_group)], start=[])
            graphs_sub.append(subgraphs)

        train_labels = train_labels.squeeze(-1)
        for pos in range(iters_per_epoch):

            batch_subgraphs = []
            for subgraphs in graphs_sub:
                selected = random.sample(subgraphs, ym_select_sub_num)
                while check_batch_graphs_nodes(selected):
                    selected = random.sample(graphs, ym_select_sub_num)
                batch_subgraphs.extend(selected)
            outputs = model(batch_subgraphs, train_input_ids, train_input_masks, train_input_types)
            loss = criterion(outputs, train_labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()

def test_final_model(model, test_graphs, test_loader, device, csv_path):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for idx, (test_input_ids, test_input_masks, test_input_types,test_labels) in enumerate(test_loader):
            test_input_ids, test_input_masks, test_input_types,test_labels = test_input_ids.to(device), test_input_masks.to(device), test_input_types.to(device), test_labels.to(device)
            

            start_idx = idx * batch_size
            end_idx = (idx + 1) * batch_size - 1
            if end_idx >= true_graph_num: end_idx = true_graph_num - 1  # 如果索引超过
            graphs = test_graphs[start_idx:end_idx + 1]
            graphs_sub = []
            for i, graph in enumerate(graphs):
                subgraphs = sum([division_func(graph, features_func, hash_func, num_group)], start=[])
                graphs_sub.append(subgraphs)

            # 转换为一维的 NumPy 数组
            test_labels = test_labels.cpu().numpy()
            y_true.extend(test_labels)  # 关键操作
            vote = [
                {
                    '0count': 0,
                    '1count': 0,
                    '0scores': 1,
                    '1scores': 0
                }
                for _ in range(batch_size)
            ]
            for pos in range(iters_per_epoch):  # 添加投票机制，至少两个预测为坏图就是坏图
                batch_subgraphs = []
                for subgraphs in graphs_sub:
                    selected = random.sample(subgraphs, ym_select_sub_num)
                    while check_batch_graphs_nodes(selected):
                        selected = random.sample(graphs, ym_select_sub_num)
                    batch_subgraphs.extend(selected)
                # 前向传播
                outputs = model(batch_subgraphs, test_input_ids, test_input_masks, test_input_types)

                # 计算预测结果
                scores = nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu().numpy()
                # 更新 vote 数组
                for i in range(batch_size):
                    if predicted[i] == 0:
                        vote[i]['0count'] += 1
                        vote[i]['0scores'] = min(vote[i]['0scores'],scores[i])
                    else:
                        vote[i]['1count'] += 1
                        vote[i]['1scores'] = max(vote[i]['1scores'],scores[i])
            for i in range(batch_size):
                if(vote[i]['1count']>=2):
                    y_pred.extend([1])
                    y_scores.extend([vote[i]['1scores']])
                else :
                    y_pred.extend([0])
                    y_scores.extend([vote[i]['0scores']])


    cm = confusion_matrix(y_true, y_pred)
    # 动态处理不同矩阵形状
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        actual_class = np.unique(y_true).item()
        if actual_class == 0:
            tn, fp, fn, tp = cm[0][0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0][0]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        raise RuntimeError(f"异常混淆矩阵形状: {cm.shape}")
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # roc_auc = roc_auc_score(y_true, y_scores)
    # 检查标签多样性
    unique_classes = np.unique(y_true)
    # 安全计算ROC AUC
    if len(unique_classes) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            print(f"无法计算ROC AUC: {str(e)}")
            roc_auc = 0
    else:
        print(f"测试集仅包含类别 {unique_classes}，跳过ROC AUC计算")
        roc_auc = 0
    if(len(unique_classes) == 2):
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(pr_recall, pr_precision)
        mcc = matthews_corrcoef(y_true, y_pred)
        y_true_flat = [item[0] for item in y_true]
        weighted_f1 = tm.F1Score(task='binary',num_classes=2, average='weighted')(torch.tensor(y_pred), torch.tensor(y_true_flat))
        weighted_f1 = weighted_f1.numpy()

        # 计算不同 FPR 下的 TPR
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        tpr_at_fpr_00001 = tpr[torch.searchsorted(torch.tensor(fpr), torch.tensor(0.0001))]
        tpr_at_fpr_0001 = tpr[torch.searchsorted(torch.tensor(fpr), torch.tensor(0.001))]
        tpr_at_fpr_001 = tpr[torch.searchsorted(torch.tensor(fpr), torch.tensor(0.01))]
        tpr_at_fpr_01 = tpr[torch.searchsorted(torch.tensor(fpr), torch.tensor(0.1))]
    else:
        precision, recall= 0,0
        pr_auc = 0
        mcc = 0
        weighted_f1 = 0
        weighted_f1 = 0
        tpr_at_fpr_00001 = 0
        tpr_at_fpr_0001 = 0
        tpr_at_fpr_001 = 0
        tpr_at_fpr_01 = 0
    # 打印结果
    print(f'TN: {tn}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print(f'TP: {tp}')
    print(f'ACC: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'ROC-AUC: {roc_auc}')
    print(f'PR-AUC: {pr_auc}')
    print(f'MCC: {mcc}')
    print(f'Weighted F1: {weighted_f1}')
    print(f'TPR@FPR=0.0001: {tpr_at_fpr_00001}')
    print(f'TPR@FPR=0.001: {tpr_at_fpr_0001}')
    print(f'TPR@FPR=0.01: {tpr_at_fpr_001}')
    print(f'TPR@FPR=0.1: {tpr_at_fpr_01}')

    # 准备保存的指标数据
    metrics = {
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'ACC': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'MCC': mcc,
        'Weighted F1': weighted_f1,
        'TPR@FPR=0.0001': tpr_at_fpr_00001,
        'TPR@FPR=0.001': tpr_at_fpr_0001,
        'TPR@FPR=0.01': tpr_at_fpr_001,
        'TPR@FPR=0.1': tpr_at_fpr_01
    }

    # 检查文件是否存在
    file_exists = os.path.isfile(csv_path)

    # 打开 CSV 文件并写入数据
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # 如果文件不存在，创建表头
            writer.writerow(metrics.keys())
        # 写入数据行
        writer.writerow(metrics.values())

    return metrics

def find_min_divisor(n,nn):
    n_nn_min = min(nn,n-nn)
    return min(n_nn_min,5)

torch.cuda.empty_cache() # 释放 GPU 缓存

csv_file_path = "path.csv"  # 替换为你的 CSV 文件路径
data = pd.read_csv(csv_file_path)
from torch_geometric.data import Data
gnndataset = []
graphs = []
true_graph_num = 0 #记录有几个有效图
bad_graph_num = 0
start_time = time.time()
for index, row in data.iterrows():
    label = row['label']  # 列名为 'label'
    html = row['HTML_Content']  # 列名为 'HTML_Content'
    # 检测 HTML_Content 是否为 NaN
    if isinstance(html, float) and math.isnan(html):
        print(f"Skipping index {index} due to NaN in HTML_Content")
        continue  # 跳过当前循环
    html = str(html)
    graph = get_graph_from_html(html,label)
    graphs.append(graph)
    data = Data(x=graph.node_features, edge_index=graph.edge_mat, y=torch.tensor([label], dtype=torch.long))
    gnndataset.append(data)
    true_graph_num = true_graph_num + 1
    if label == 1: bad_graph_num = bad_graph_num + 1

    torch.cuda.empty_cache() # 释放 GPU 缓存
print(f"total={true_graph_num},bad={bad_graph_num}")
fold_n = find_min_divisor(true_graph_num,bad_graph_num)
input_ids = []  # input char ids
input_types = []  # segment ids
input_masks = []  # attention mask
label = []
dataprocess(csv_file_path, input_ids, input_types, input_masks, label)
#划分数据集
train_graphs, test_graphs, train_input_ids, test_input_ids, train_input_types, test_input_types, \
train_input_masks, test_input_masks, train_labels, test_labels,train_gnndataset,test_gnndataset\
    =separate_data(graphs, input_ids, input_types, input_masks, label, gnndataset,seed, fold_idx, fold_n)

train_data = TensorDataset(torch.tensor(train_input_ids).to(device),
                               torch.tensor(train_input_masks).to(device),
                               torch.tensor(train_input_types).to(device),
                               torch.tensor(train_labels).to(device))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(torch.tensor(test_input_ids).to(device),
                             torch.tensor(test_input_masks).to(device),
                             torch.tensor(test_input_types).to(device),
                             torch.tensor(test_labels).to(device))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

#加载模型
input_dim = train_graphs[0].node_features.shape[1]
gnn_model = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                        num_classes, final_dropout, learn_eps, graph_pooling_type,
                        neighbor_pooling_type, device).to(device)
bert_model = myConvBertModel().to(device)
model = FinalModel(gnn_model, bert_model, gnn_hidden_dim=768, bert_hidden_dim=768, num_classes=2)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
result_save_path = 'result.csv'


num_features = gnndataset[0].num_node_features if gnndataset else None
# 3. 数据加载器
train_gnn_loader = DataLoader(train_gnndataset, batch_size=batch_size)
test_gnn_loader = DataLoader(test_gnndataset, batch_size=batch_size)
# 初始化模型
gcn_model = GCN(num_features, num_classes).to(device)
gin_model = GIN(num_features, num_classes).to(device)
grn_model = GRN(num_features, num_classes).to(device)
# 初始化优化器
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=learning_rate)
gin_optimizer = torch.optim.Adam(gin_model.parameters(), lr=learning_rate)
grn_optimizer = torch.optim.Adam(grn_model.parameters(), lr=learning_rate)

# 开始训练
best = 0
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    loss = train_final_model(model, train_graphs, train_loader, criterion, optimizer, device)
    metrics = test_final_model(model, test_graphs, test_loader, device, result_save_path)
    if(metrics['ACC']>best):
        best=metrics['ACC']
        torch.save(model.state_dict(),'best_final_model.pth')

    torch.cuda.empty_cache()
    loss = gnn_train(gcn_model, gcn_optimizer, train_gnn_loader, device, epoch)
    gnn_evaluate(gcn_model, test_gnn_loader, device, epoch, csv_file='gcn_metrics.csv')

    torch.cuda.empty_cache()
    loss = gnn_train(gin_model, gin_optimizer, train_gnn_loader, device, epoch)
    gnn_evaluate(gin_model, test_gnn_loader, device, epoch, csv_file='gin_etrics.csv')

    torch.cuda.empty_cache()
    loss = gnn_train(grn_model, grn_optimizer, train_gnn_loader, device, epoch)
    gnn_evaluate(grn_model, test_gnn_loader, device, epoch, csv_file='grn_etrics.csv')
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
