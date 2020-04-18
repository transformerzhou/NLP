#! encoding:utf-8

import tensorflow as tf
import numpy as np
import os
from scipy import sparse
from collections import defaultdict


class TuckERLoader():
    def __init__(self, base_path="data", reverse=True):
        self.train_data = self.load_data(base_path, 'train.txt', reverse=reverse)
        self.valid_data = self.load_data(base_path, 'valid.txt', reverse=reverse)
        self.test_data = self.load_data(base_path, 'test.txt', reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)

        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)

        self.relations = self.train_relations + \
                         [i for i in self.valid_relations if i not in self.train_relations] + \
                         [i for i in self.test_relations if i not in self.train_relations]

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.realtion_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        del self.train_relations
        del self.test_relations
        del self.valid_relations

    def data_dump(self, data="train"):
        if data == "train":
            data = self.train_data
        elif data == "valid":
            data = self.valid_data
        elif data == "test":
            data = self.test_data
        data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        return er_vocab, er_vocab_pairs

    def load_data(self, base_path, data_type="train", reverse=False):
        data = []
        with open(os.path.join(base_path, data_type)) as rf:
            for line in rf:
                contents = line.strip().split()
                data.append(contents)
                if reverse:
                    data.append([contents[2], contents[1] + "_reverse", contents[0]])
            return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def get_batch(self, er_vocab, er_vocab_pairs, batch_size=32):

        #
        targets = [er_vocab[key] for key in er_vocab_pairs]

        def generator():
            for x, y in zip(er_vocab_pairs, targets):
                yield {'h_r': x, 't': y}

        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types={'h_r': tf.int32, 't': tf.int32})
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes={'h_r': [None], 't': [None]},
                                       drop_remainder=True)
        return dataset

    def get_data_idxs(self, data):  # data could be self.train_data self.valid_data self.test_data
        data_idxs = [
            (self.entity_idxs[data[i][0]], self.realtion_idxs[data[i][1]],
             self.entity_idxs[data[i][2]]) for i in range(len(data))
        ]
        print("Number of data points: %d" % len(data_idxs))
        return data_idxs

    def get_er_vocab(self, data_index):
        er_vocab = defaultdict(list)
        for triple in data_index:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def target_convert(self, targets, batch_size, num_entities):
        targets_one_hot = np.zeros((batch_size, num_entities))
        for idx, tar in enumerate(targets):
            targets_one_hot[idx, tf.gather_nd(tar, tf.where(tar > 0))] = 1
        return tf.constant(targets_one_hot)


class GCNLoader():
    def __init__(self, base_path="data", dataset="cora"):
        self.base_path = base_path
        self.dataset = dataset
        print("Loading {} dataset...".format(dataset))

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = (1 / rowsum).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sparse.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def convert_2_sparse_tensor(self, sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
        values = sparse_matrix.data
        shape = sparse_matrix.shape
        indices = np.array([[row, col] for row, col in zip(sparse_matrix.row, sparse_matrix.col)], dtype=np.int64)
        return tf.sparse.SparseTensor(indices, values, shape)

    def load(self):
        idx_features_labels = np.genfromtxt("{}/{}.content".format(self.base_path, self.dataset),
                                            dtype=np.dtype(str))
        features = sparse.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # 构建图
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/{}.cites".format(self.base_path, self.dataset),
                                        dtype=np.int32)
        # [[1,2],
        #  [22,23]]
        # N*2
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)

        # 构建对称邻接矩阵
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize(features)
        adj = self.normalize(adj + sparse.eye(adj.shape[0]))

        features = tf.constant(np.array(features.todense()))

        labels = tf.constant(np.where(labels)[1])
        adj = self.convert_2_sparse_tensor(adj)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        return features, adj, labels, idx_train, idx_val, idx_test


class RGCNLoader(object):
    def __init__(self, base_path="data", dataset="FB15k"):
        self.base_path = base_path
        self.dataset = dataset
        self.file_path = os.path.join(base_path, dataset)

    def read_triplets(self, file_path, entity2id, relation2id):
        triplets = []

        with open(file_path) as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

        return np.array(triplets)

    def load_data(self):
        print("load data from {}".format(self.file_path))

        with open(os.path.join(self.file_path, 'entities.dict')) as f:
            entity2id = dict()

            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(self.file_path, 'relations.dict')) as f:
            relation2id = dict()

            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_triplets = self.read_triplets(os.path.join(self.file_path, 'train.txt'), entity2id, relation2id)
        valid_triplets = self.read_triplets(os.path.join(self.file_path, 'valid.txt'), entity2id, relation2id)
        test_triplets = self.read_triplets(os.path.join(self.file_path, 'test.txt'), entity2id, relation2id)

        print('num_entity: {}'.format(len(entity2id)))
        print('num_relation: {}'.format(len(relation2id)))
        print('num_train_triples: {}'.format(len(train_triplets)))
        print('num_valid_triples: {}'.format(len(valid_triplets)))
        print('num_test_triples: {}'.format(len(test_triplets)))

        return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

    def sample_edge_uniform(self, n_triples, sample_size):
        """Sample edges uniformly from all the edges."""
        all_edges = np.arange(n_triples)
        return np.random.choice(all_edges, sample_size, replace=False)

    def negative_sampling(self, pos_samples, num_entity, negative_rate):
        size_of_batch = len(pos_samples)
        num_to_generate = size_of_batch * negative_rate
        neg_samples = np.tile(pos_samples, (negative_rate, 1))
        labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
        labels[: size_of_batch] = 1
        values = np.random.choice(num_entity, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        return np.concatenate((pos_samples, neg_samples)), labels

    def edge_normalization(self, edge_type, edge_index, num_entity, num_relation):
        from fennlp.gnn.scatter import scatter_sum
        '''
        
            Edge normalization trick
            - one_hot: (num_edge, num_relation)
            - deg: (num_node, num_relation)
            - index: (num_edge)
            - deg[edge_index[0]]: (num_edge, num_relation)
            - edge_norm: (num_edge)
        '''
        one_hot = tf.one_hot(tf.cast(edge_type, np.int32),
                             2 * num_relation, dtype=tf.int64)
        one_hot = tf.constant(one_hot.numpy())
        deg = scatter_sum(one_hot, edge_index[0], dim=0, dim_size=num_entity)
        index = edge_type + tf.keras.backend.arange(len(edge_index[0])) * (2 * num_relation)
        edge_norm = 1 / np.reshape(deg[edge_index[0]], -1)[index]

        return edge_norm

    def generate_sampled_graph_and_labels(self, triplets, batch_size, split_size, num_entity, num_rels, negative_rate):
        """
            Get training graph and signals
            First perform edge neighborhood sampling on graph, then perform negative
            sampling to generate negative samples
        """

        edges = self.sample_edge_uniform(len(triplets), batch_size)

        # Select sampled edges
        edges = triplets[edges]
        src, rel, dst = edges.transpose()
        uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # Negative sampling
        samples, labels = self.negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)
        # samples 是所有的三元组，labels是表示该三元组是真是假
        # further split graph, only half of the edges will be used as graph
        # structure, while the rest half is used as unseen positive samples
        split_size = int(batch_size * split_size)
        graph_split_ids = np.random.choice(np.arange(batch_size),
                                           size=split_size, replace=False)

        src = tf.constant(src[graph_split_ids], dtype=tf.float32)
        dst = tf.constant(dst[graph_split_ids], dtype=tf.float32)
        rel = tf.constant(rel[graph_split_ids], dtype=tf.float32)
        # Create bi-directional graph
        src, dst = tf.concat((src, dst), axis=0), tf.concat((dst, src), axis=0)
        rel = tf.concat((rel, rel + num_rels), axis=0)
        edge_type = rel
        self.edge_index = tf.stack((src, dst))
        self.entity = tf.constant(uniq_entity)
        self.edge_type = edge_type
        self.edge_norm = tf.ones(edge_type.shape)

        self.samples = tf.constant(samples)
        self.labels = tf.constant(labels)
