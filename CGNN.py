#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = "Liuzg"

import tensorflow as tf
from utils import *
from sklearn import preprocessing
import scipy.sparse as sp
import os

from xgboost import XGBClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


def tonohot(onehot):
    nohot = [np.argmax(item) for item in onehot]
    return nohot


def all_diffs(a, b):
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def euclidean_dist(embed1, embed2):
    diffs = all_diffs(embed1, embed2)
    return tf.sqrt(tf.reduce_sum(input_tensor=tf.square(diffs), axis=-1) + 1e-12)


def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=tf.stop_gradient(labels))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    loss *= mask
    return tf.reduce_mean(input_tensor=loss)


def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=labels, axis=1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    accuracy_all *= mask
    return tf.reduce_mean(input_tensor=accuracy_all)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def tf_normalize_adj(inputs):
    params_shape = tf.shape(inputs)[0]
    inputs = tf.cast(inputs, dtype=tf.float32) + tf.cast(tf.eye(params_shape), dtype=tf.float32)
    row_sum = tf.reduce_sum(inputs, axis=1)
    d_inv_sqrt = tf.pow(row_sum, -0.5)
    tf.clip_by_value(d_inv_sqrt, clip_value_min=0.01, clip_value_max=1000)
    outputs = tf.transpose(d_inv_sqrt * inputs) * d_inv_sqrt
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)


def super_cast(*args, dtype):
    cast_list = []
    for arg in args:
        arg = tf.cast(arg, dtype=dtype)
        cast_list.append(arg)
    return cast_list


class PredictModel(tf.keras.Model):

    def __init__(self, num_classes, y_train_val, K, processed_adj, feature_weight, rate=0.5):
        super(PredictModel, self).__init__()
        self.regularizer = tf.keras.regularizers.l2(l=0.5 * 5e-4)
        self.y_train_val = y_train_val
        self.K = tf.constant(K)
        self.processed_adj0 = tf.constant(processed_adj)
        self.processed_adj1 = tf.constant(processed_adj)
        self.sim_fw = tf.Variable(feature_weight, dtype=tf.float32, name='sim_fw', trainable=True)

        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dense64 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, use_bias=False,
                                             kernel_regularizer=self.regularizer)
        self.dense_logits = tf.keras.layers.Dense(units=num_classes, activation=tf.nn.sigmoid, use_bias=False,
                                                  kernel_regularizer=self.regularizer)

    def call(self, inputs, training=None):
        x = self.sim_fw * inputs
        x = self.dropout1(x)
        x = self.dense64(x)
        x = self.processed_adj0 @ tf.cast(x, dtype=tf.float32)
        logits = self.dense_logits(x)
        return logits


class Model(tf.keras.Model):
    def __init__(self, features, labels, feature_dense_y, num_classes, y_train_val, K, train_mask, wc, wd, wi, warmup_sim_W_flag=0, rate=0.5, Index='expectedgain', NameNo=0, tree_max_depth=3):
        super(Model, self).__init__()
        self.features = features
        self.wc, self.wd, self.wi = wc, wd, wi
        self.labels = labels
        self.y_train_val = y_train_val
        self.K = K
        self.train_mask = train_mask
        self.Index = Index
        self.warmup_sim_W = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.warmup_sim_W_flag = tf.constant(warmup_sim_W_flag, dtype=tf.float32)
        graph_input_feature_len = feature_dense_y.shape[1]
        disturb = tf.random.uniform([1, graph_input_feature_len], 0.0, 1e-4) + tf.ones([1, graph_input_feature_len], dtype=tf.float32)

        self.sim_W = tf.Variable(disturb, dtype=tf.float32, name=f'sim_w{NameNo}', trainable=True)
        self.features, self.importance_list = self.FeaProcess(self.features, fk=10, max_depth=tree_max_depth)
        feature_weight = self.FeatureWeightAuto(self.features)
        self.sim_fw = tf.Variable(initial_value=feature_weight, dtype=tf.float32, name=f'sim_fw{NameNo}', trainable=True)

        self.regularizer = tf.keras.regularizers.l2(l=0.5 * 5e-4)
        self.dropout1 = tf.keras.layers.Dropout(rate=rate)

        self.dense64 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, use_bias=False,
                                             kernel_regularizer=self.regularizer)

        self.dense_logits = tf.keras.layers.Dense(units=num_classes, activation=tf.nn.sigmoid, use_bias=False,
                                                  kernel_regularizer=self.regularizer)

    def call(self, feature_dense_y, training=None):
        feature_dense_y_w = feature_dense_y * self.sim_W
        sim_matrix = feature_dense_y_w @ tf.transpose(feature_dense_y_w)
        top_values, top_indices = tf.nn.top_k(sim_matrix, k=self.K)
        kthvalue = tf.math.reduce_min(top_values, axis=-1)
        adj = tf.cast((sim_matrix > kthvalue), tf.float32) * sim_matrix
        self.processed_adj0 = tf_normalize_adj(adj)
        lap_mul_input = self.processed_adj0 @ self.features
        lap_mul_input = self.sim_fw * lap_mul_input

        x = lap_mul_input
        x = self.dropout1(x)
        x = self.dense64(x)

        dense_1_y = tf.concat([x, tf.cast(tf.convert_to_tensor(self.y_train_val), dtype=tf.float32)], 1)
        sim_matrix = dense_1_y @ tf.transpose(dense_1_y)
        top_values, top_indices = tf.nn.top_k(sim_matrix, k=self.K)
        kthvalue = tf.math.reduce_min(top_values, axis=-1)
        adj = tf.cast(tf.cast((sim_matrix > kthvalue), tf.int32), tf.float32) * sim_matrix
        adj = tf.cast(adj, dtype=tf.float32)
        self.processed_adj1 = tf_normalize_adj(adj)
        x = self.processed_adj0 @ tf.cast(x, dtype=tf.float32)
        logits = self.dense_logits(x)
        return logits

    def FeaProcess(self, features, fk=10, max_depth=20):
        X, y = features[self.train_mask], tonohot(self.labels[self.train_mask])
        feature_num = X.shape[1]
        dir_con_list = [set({fno}) for fno in range(feature_num)]
        indir_con_list = [set({fno}) for fno in range(feature_num)]
        import xgboost as xgb
        objective = 'multi:softmax'
        n_class = len(np.unique(y))
        if n_class == 2:
            objective = "binary:logistic"
        clf = XGBClassifier(learning_rate=0.11, max_depth=max_depth, n_estimators=1000, objective=objective, n_jobs=int(os.cpu_count() // 1.2))
        clf.fit(X, y)
        booster = clf.get_booster()
        ensemble_trees = xgb.Booster.get_dump(booster, with_stats=True)
        Interactions_dict = GetStatistics(booster, ensemble_trees, MaxInteractionDepth=10, SortBy=self.Index)
        Interaction, AverageTreeDepth = GetAverageTreeDepth(Interactions_dict, Depth=f'Depth{1}')
        for relation in Interaction:
            r = list(map(int, relation.replace('f', '').split('|')))
            for j in range(len(r)):
                front_part, back_part = r[0: j], r[j + 1:]
                for fp in front_part:
                    dir_con_list[r[j]].add(fp)
                for bp in back_part:
                    dir_con_list[r[j]].add(bp)
        for i in range(2, fk):
            Depth = f'Depth{i}'
            Interaction, AverageTreeDepth = GetAverageTreeDepth(Interactions_dict, Depth=Depth)
            for relation in Interaction:
                r = list(map(int, relation.replace('f', '').split('|')))
                for j in range(len(r)):
                    front_part, back_part = r[0: j], r[j + 1:]
                    for fp in front_part:
                        if fp not in dir_con_list[r[j]]:
                            indir_con_list[r[j]].add(fp)
                    for bp in back_part:
                        if bp not in dir_con_list[r[j]]:
                            indir_con_list[r[j]].add(bp)
            if Interaction == [] or AverageTreeDepth == []:
                break
        features = self.FeaCon(features, dir_con_list, indir_con_list, self.wc, self.wd, self.wi)
        X = features[self.train_mask]
        clf.fit(X, y)
        booster = clf.get_booster()
        ensemble_trees = xgb.Booster.get_dump(booster, with_stats=True)
        Interactions_dict = GetStatistics(booster, ensemble_trees, MaxInteractionDepth=10, SortBy=self.Index)
        importance_list = GetImportanceFeature(Interactions_dict, Depth=f'Depth{0}', Index=self.Index)
        return features, importance_list

    def FeatureWeightAuto(self, lap_mul_input):
        feature_weight = np.ones(lap_mul_input.shape[1], dtype=np.float32)
        total = sum([f.importanceValue for f in self.importance_list])
        for f in self.importance_list:
            feature_weight[f.feature] *= np.e ** (f.importanceValue / total)
        return tf.convert_to_tensor(feature_weight)

    @staticmethod
    def FeaCon(X, dir_con_list, indir_con_list, wc, wd, wi):
        feature_num = X.shape[1]
        X = np.array(X)
        new_X = np.zeros(X.shape, dtype=float)
        for fi in range(feature_num):
            if len(dir_con_list[fi]) > 1:
                c2 = len(dir_con_list[fi]) - 1
                new_X[:, fi] = wc * X[:, fi] + sum(wd * X[:, j] for j in dir_con_list[fi] if j != fi)
                c1 = 0
                if len(indir_con_list[fi]) > 1:
                    c1 = len(indir_con_list[fi]) - 1
                    new_X[:, fi] = new_X[:, fi] + sum(wi * X[:, j] for j in dir_con_list[fi] if j != fi)
                t = 1 + c2 + c1
                if wc + wd * c2 + wi * c1 > t:
                    new_X[:, fi] = (1 / t) * new_X[:, fi]
            else:
                new_X[:, fi] = X[:, fi]
        return new_X


from mxgbfir import GetImportanceFeature, GetAverageTreeDepth, GetStatistics


def FeaturesChange(features, use_label=None, times=1):
    import scipy as sp
    if isinstance(features, sp.sparse.csr.csr_matrix):
        features = features.todense()
    if use_label is not None:
        y = use_label
        while times > 1:
            y = np.concatenate((y, use_label), axis=1)
            times -= 1
        feature_dense_y = tf.cast(np.concatenate((y, features), axis=1), dtype=tf.float32)
    else:
        feature_dense_y = tf.cast(features, dtype=tf.float32)
    return features, feature_dense_y


def GetLoss(preds, labels, mask, regularizer_loss, adj_loss=None):
    classification_loss = masked_softmax_cross_entropy(preds, labels, mask)
    loss = classification_loss + regularizer_loss
    if adj_loss is not None:
        loss = classification_loss + regularizer_loss + adj_loss
    return loss


def LapChange(features, feature_dense_y, sim_W, K=10):
    feature_dense_y_w = feature_dense_y * sim_W
    sim_matrix = feature_dense_y_w @ tf.transpose(feature_dense_y_w)
    top_values, top_indices = tf.nn.top_k(sim_matrix, k=K)
    kthvalue = tf.math.reduce_min(top_values, axis=-1)
    adj = tf.cast((sim_matrix > kthvalue), tf.float32) * sim_matrix
    processed_adj = tf_normalize_adj(adj)
    lap_mul_input = processed_adj @ features
    return lap_mul_input, processed_adj


def FeaProcess(features, labels, mask, max_depth=10, k=10, SortBy='expectedgain'):
    X, y = features[mask], tonohot(labels[mask])
    feature_num = X.shape[1]
    dir_con_list = [set({fno}) for fno in range(feature_num)]
    indir_con_list = [set({fno}) for fno in range(feature_num)]
    import xgboost as xgb
    objective = 'multi:softmax'
    n_class = len(np.unique(y))
    if n_class == 2:
        objective = "binary:logistic"
    clf = XGBClassifier(learning_rate=0.11, max_depth=max_depth, n_estimators=1000, objective=objective, n_jobs=int(os.cpu_count() // 1.2))
    clf.fit(X.numpy(), y)
    booster = clf.get_booster()
    ensemble_trees = xgb.Booster.get_dump(booster, with_stats=True)
    Interactions_dict = GetStatistics(booster, ensemble_trees, MaxInteractionDepth=10, SortBy=SortBy)
    importance_list = GetImportanceFeature(Interactions_dict, Depth=f'Depth{0}', Index=SortBy)
    Interaction, AverageTreeDepth = GetAverageTreeDepth(Interactions_dict, Depth=f'Depth{1}')
    for relation in Interaction:
        r = list(map(int, relation.replace('f', '').split('|')))
        for j in range(len(r)):
            front_part, back_part = r[0: j], r[j + 1:]
            for fp in front_part:
                dir_con_list[r[j]].add(fp)
            for bp in back_part:
                dir_con_list[r[j]].add(bp)
    for i in range(2, k):
        Depth = f'Depth{i}'
        Interaction, AverageTreeDepth = GetAverageTreeDepth(Interactions_dict, Depth=Depth)
        for relation in Interaction:
            r = list(map(int, relation.replace('f', '').split('|')))
            for j in range(len(r)):
                front_part, back_part = r[0: j], r[j + 1:]
                for fp in front_part:
                    if fp not in dir_con_list[r[j]]:
                        indir_con_list[r[j]].add(fp)
                for bp in back_part:
                    if bp not in dir_con_list[r[j]]:
                        indir_con_list[r[j]].add(bp)
        if Interaction == [] or AverageTreeDepth == []:
            break
    return importance_list, dir_con_list, indir_con_list


def FeaCon(X, dir_con_list, indir_con_list, wc, wd, wi):
    feature_num = X.shape[1]
    X = X.numpy()
    new_X = np.zeros(X.shape, dtype=float)
    for fi in range(feature_num):
        if len(dir_con_list[fi]) > 1:
            c2 = len(dir_con_list[fi]) - 1
            new_X[:, fi] = wc * X[:, fi] + sum(wd * X[:, j] for j in dir_con_list[fi] if j != fi)
            c1 = 0
            if len(indir_con_list[fi]) > 1:
                c1 = len(indir_con_list[fi]) - 1
                new_X[:, fi] = new_X[:, fi] + sum(wi * X[:, j] for j in dir_con_list[fi] if j != fi)
            t = 1 + c2 + c1
            if wc + wd * c2 + wi * c1 > t:
                new_X[:, fi] = (1 / t) * new_X[:, fi]
        else:
            new_X[:, fi] = X[:, fi]
    return tf.convert_to_tensor(new_X)


def FeatureWeightAuto(lap_mul_input, importance_list):
    feature_weight = np.ones(lap_mul_input.shape[1], dtype=np.float32)
    total = sum([f.importanceValue for f in importance_list])
    for f in importance_list:
        feature_weight[f.feature] *= np.e ** (f.importanceValue / total)
    return tf.convert_to_tensor(feature_weight)


def main(optmaizer_learning_rate, predict_learning_rate, optmaizerEPOCHS, predictEPOCHS, DataName, K=10, wc=1, wd=1, wi=1, warmup_sim_W_flag=0):
    if DataName in ('cora', 'citeseer', 'pubmed'):
        ori_adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(DataName)
        ori_adj = ori_adj.todense().astype(np.float32)
    else:
        from utils import Datas
        uciconfig = {
            'wine': {'n_train': 10, 'n_val': 20, 'n_test': 158, 'scale': True},
            'cancer': {'n_train': 10, 'n_val': 20, 'n_test': 539, 'scale': True},
            'digits': {'n_train': 50, 'n_val': 100, 'n_test': 1647, 'scale': False},
            '20news10': {'n_train': 100, 'n_val': 200, 'n_test': 9307, 'scale': False},
            'fma': {'n_train': 160, 'n_val': 320, 'n_test': 7514, 'scale': False}
        }
        uci = Datas(DataName, n_train=uciconfig.get(DataName).get('n_train'),
                    n_val=uciconfig.get(DataName).get('n_val'),
                    n_es=-1, scale=uciconfig.get(DataName).get('scale'))
        _, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = uci.load()

    features = preprocessing.normalize(features)

    num_classes = y_train.shape[1]
    y_train_val = y_train + y_val

    features, feature_dense_y = FeaturesChange(features, use_label=y_train_val, times=1)

    SortByList = ['expectedgain', 'averagegain', 'averagecover']
    submodels = []
    for i in range(len(SortByList)):
        submodel = Model(features, labels, feature_dense_y, num_classes, y_train_val, K, train_mask, wc, wd, wi, warmup_sim_W_flag=warmup_sim_W_flag, rate=0.5, Index=SortByList[i], NameNo=i, tree_max_depth=i+2)
        submodels.append(submodel)
    feature_dense_y_list = [feature_dense_y] * len(submodels)

    train_optimizer = tf.keras.optimizers.Adamax(learning_rate=optmaizer_learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    simWs = [0.] * (len(submodels))

    @tf.function
    def train_step(inputs, labels):
        outputs_list, loss_all_list, no_sim_W_list = [], [], []
        with tf.GradientTape(persistent=True) as tape:
            for j in range(len(submodels)):
                outputs_list.append(submodels[j](inputs[j], training=True))
                loss = GetLoss(outputs_list[j], labels, train_mask, submodels[j].losses, tf.reduce_mean(tf.abs(submodels[j].processed_adj1 - submodels[j].processed_adj0)))
                loss_all_list.append(loss)
                temp_no_sim_W_list = []
                for v in submodels[j].trainable_variables:
                    if v.name.find('sim_w') == -1:
                        temp_no_sim_W_list.append(v)
                no_sim_W_list.append(temp_no_sim_W_list)
            outputs, loss_all = outputs_list[0], loss_all_list[0]
            for j in range(1, len(submodels)):
                loss_all += loss_all_list[j]
                outputs += outputs_list[j]
            loss_all = loss_all / len(submodels)

        for j in range(len(submodels)):
            if submodels[j].warmup_sim_W < submodels[j].warmup_sim_W_flag:
                gradients = tape.gradient(loss_all, no_sim_W_list[j])
                train_optimizer.apply_gradients(zip(gradients, no_sim_W_list[j]))
            else:
                gradients = tape.gradient(loss_all, submodels[j].trainable_variables)
                train_optimizer.apply_gradients(zip(gradients, submodels[j].trainable_variables))
            simWs[j] = submodels[j].sim_W

        train_loss(loss_all)
        train_accuracy(masked_accuracy(outputs, labels, train_mask))

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

    @tf.function
    def val_step(inputs, labels):
        outputs_list, loss_all_list = [], []
        for j in range(len(submodels)):
            outputs_list.append(submodels[j](inputs[j], training=False))
            loss = GetLoss(outputs_list[j], labels, val_mask, submodels[j].losses, tf.reduce_mean(tf.abs(submodels[j].processed_adj1 - submodels[j].processed_adj0)))
            loss_all_list.append(loss)
        outputs, loss_all = outputs_list[0], loss_all_list[0]
        for j in range(1, len(submodels)):
            loss_all += loss_all_list[j]
            outputs += outputs_list[j]
        loss_all = loss_all / len(submodels)

        val_loss(loss_all)
        val_accuracy(masked_accuracy(outputs, labels, val_mask))

    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

    @tf.function
    def test_step(inputs, labels):
        outputs_list = []
        for j in range(len(submodels)):
            outputs_list.append(submodels[j](inputs[j], training=False))
        outputs = outputs_list[0]
        for j in range(1, len(submodels)):
            outputs += outputs_list[j]
        test_accuracy(masked_accuracy(outputs, labels, test_mask))
        return tf.argmax(input=outputs, axis=1)

    for epoch in range(optmaizerEPOCHS):
        train_loss.reset_states(), train_accuracy.reset_states()
        val_loss.reset_states(), val_accuracy.reset_states()
        train_step(feature_dense_y_list, y_train)
        val_step(feature_dense_y_list, y_val)
        test_step(feature_dense_y_list, y_test)
        for j in range(len(submodels)):
            submodels[j].warmup_sim_W.assign_add(1)

    predict_submodels = []
    lap_mul_input_list, processed_adj_list = [None] * len(simWs), [None] * len(simWs)

    for i in range(len(simWs)):
        lap_mul_input_list[i], processed_adj_list[i] = LapChange(features, feature_dense_y, simWs[i], K)
        dir_con_list, indir_con_list = FeaProcess(lap_mul_input_list[i], y_train, train_mask, max_depth=i+2, k=10, SortBy=SortByList[i])[1:]
        lap_mul_input_list[i] = FeaCon(lap_mul_input_list[i], dir_con_list, indir_con_list, wc, wd, wi)
        importance_list = FeaProcess(lap_mul_input_list[i], y_train, train_mask, max_depth=i+2, k=10, SortBy=SortByList[i])[0]
        feature_weight = FeatureWeightAuto(lap_mul_input_list[i], importance_list)
        predict_submodel = PredictModel(num_classes, y_train_val, K, processed_adj_list[i], feature_weight, rate=0.5)
        predict_submodels.append(predict_submodel)
    predict_optimizer = tf.keras.optimizers.Adamax(learning_rate=predict_learning_rate)

    @tf.function
    def perdict_train_step(inputs, labels):
        outputs_list, loss_all_list = [], []
        with tf.GradientTape(persistent=True) as tape:
            for j in range(len(predict_submodels)):
                outputs_list.append(predict_submodels[j](inputs[j], training=True))
                loss = GetLoss(outputs_list[j], labels, train_mask, predict_submodels[j].losses, None)
                loss_all_list.append(loss)
            outputs, loss_all = outputs_list[0], loss_all_list[0]
            for j in range(1, len(predict_submodels)):
                loss_all += loss_all_list[j]
                outputs += outputs_list[j]
            loss_all = loss_all / len(predict_submodels)

        for j in range(len(predict_submodels)):
            gradients = tape.gradient(loss_all, predict_submodels[j].trainable_variables)
            predict_optimizer.apply_gradients(zip(gradients, predict_submodels[j].trainable_variables))

        train_loss(loss_all)
        train_accuracy(masked_accuracy(outputs, labels, train_mask))

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

    @tf.function
    def perdict_val_step(inputs, labels):
        outputs_list, loss_all_list = [], []
        for j in range(len(predict_submodels)):
            outputs_list.append(predict_submodels[j](inputs[j], training=False))
            loss = GetLoss(outputs_list[j], labels, train_mask, predict_submodels[j].losses, None)
            loss_all_list.append(loss)
        outputs, loss_all = outputs_list[0], loss_all_list[0]
        for j in range(1, len(predict_submodels)):
            loss_all += loss_all_list[j]
            outputs += outputs_list[j]
        loss_all = loss_all / len(predict_submodels)

        val_loss(loss_all)
        val_accuracy(masked_accuracy(outputs, labels, val_mask))

    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

    @tf.function
    def perdict_test_step(inputs, labels):
        outputs_list = []
        for j in range(len(predict_submodels)):
            outputs_list.append(predict_submodels[j](inputs[j], training=False))
        outputs = outputs_list[0]
        for j in range(1, len(predict_submodels)):
            outputs += outputs_list[j]
        test_accuracy(masked_accuracy(outputs, labels, test_mask))
        return tf.argmax(input=outputs, axis=1)

    test_accuracy_max = 0.
    opt_epoch = 0
    for epoch in range(predictEPOCHS):
        train_loss.reset_states(), train_accuracy.reset_states()
        val_loss.reset_states(), val_accuracy.reset_states()
        test_accuracy.reset_states()

        perdict_train_step(lap_mul_input_list, y_train)
        perdict_val_step(lap_mul_input_list, y_val)
        perdict_test_step(lap_mul_input_list, y_test)
        if test_accuracy_max < test_accuracy.result():
            test_accuracy_max = test_accuracy.result()
            opt_epoch = epoch
    CA = test_accuracy.result()
    print(f'Acc={CA * 100}')

def Config():
    DataName = 'wine'
    optmaizer_learning_rate = 0.01
    predict_learning_rate = 0.003
    optmaizerEPOCHS = 500
    warmup_sim_W_flag = 300
    predictEPOCHS = 800
    K = 50
    wc, wd, wi = 1.2, 1.1, 1
    return DataName, optmaizer_learning_rate, predict_learning_rate, optmaizerEPOCHS, warmup_sim_W_flag, predictEPOCHS, K, wc, wd, wi


if __name__ == '__main__':
    DataName, optmaizer_learning_rate, predict_learning_rate, optmaizerEPOCHS, warmup_sim_W_flag, predictEPOCHS, K, wc, wd, wi = Config()
    main(optmaizer_learning_rate, predict_learning_rate, optmaizerEPOCHS, predictEPOCHS, DataName, K=K, wc=wc, wd=wd, wi=wi, warmup_sim_W_flag=warmup_sim_W_flag)