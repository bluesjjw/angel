/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */


package com.tencent.angel.ml.core.conf

import com.tencent.angel.ml.matrix.RowType
import com.tencent.angel.model.output.format.{ColIdValueTextRowFormat, RowIdColIdValueTextRowFormat, TextColumnFormat}

object MLConf {

  // The Train action means to learn a model while Predict action uses the model to
  // predict beyond unobserved samples.
  val ANGEL_ML_TRAIN = "train"
  val ANGEL_ML_PREDICT = "predict"
  val ANGEL_ML_INC_TRAIN = "inctrain"

  // Data params
  val ML_DATA_INPUT_FORMAT = "ml.data.type"
  val DEFAULT_ML_DATA_INPUT_FORMAT = "libsvm"
  val ML_DATA_SPLITOR = "ml.data.splitor"
  val DEFAULT_ML_DATA_SPLITOR = "\\s+"
  val ML_DATA_IS_NEGY = "ml.data.is.negy"
  val DEFAULT_ML_DATA_IS_NEGY = true
  val ML_DATA_HAS_LABEL = "ml.data.has.label"
  val DEFAULT_ML_DATA_HAS_LABEL = true
  val ML_DATA_LABEL_TRANS = "ml.data.label.trans.class"
  val DEFAULT_ML_DATA_LABEL_TRANS = "NoTrans"
  val ML_DATA_LABEL_TRANS_THRESHOLD = "ml.data.label.trans.threshold"
  val DEFAULT_ML_DATA_LABEL_TRANS_THRESHOLD = 0
  val ML_VALIDATE_RATIO = "ml.data.validate.ratio"
  val DEFAULT_ML_VALIDATE_RATIO = 0.05
  val ML_FEATURE_INDEX_RANGE = "ml.feature.index.range"
  val DEFAULT_ML_FEATURE_INDEX_RANGE = -1
  val ML_BLOCK_SIZE = "ml.block.size"
  val DEFAULT_ML_BLOCK_SIZE = 1000000
  val ML_DATA_USE_SHUFFLE = "ml.data.use.shuffle"
  val DEFAULT_ML_DATA_USE_SHUFFLE = false
  val ML_DATA_POSNEG_RATIO = "ml.data.posneg.ratio"
  val DEFAULT_ML_DATA_POSNEG_RATIO = -1


  // Worker params
  val ANGEL_WORKER_THREAD_NUM = "angel.worker.thread.num"
  val DEFAULT_ANGEL_WORKER_THREAD_NUM = 1

  // network param
  val ANGEL_COMPRESS_BYTES = "angel.compress.bytes"
  val DEFAULT_ANGEL_COMPRESS_BYTES = 8

  // Model params
  val ML_MODEL_CLASS_NAME = "ml.model.class.name"
  val DEFAULT_ML_MODEL_CLASS_NAME = ""
  val ML_MODEL_SIZE = "ml.model.size"
  val DEFAULT_ML_MODEL_SIZE = -1
  val ML_MODEL_TYPE = "ml.model.type"
  val DEFAULT_ML_MODEL_TYPE = RowType.T_FLOAT_DENSE.toString
  val ML_MODEL_IS_CLASSIFICATION = "ml.model.is.classification"
  val DEFAULT_ML_MODEL_IS_CLASSIFICATION = true

  val ML_EPOCH_NUM = "ml.epoch.num"
  val DEFAULT_ML_EPOCH_NUM = 30
  val ML_BATCH_SAMPLE_RATIO = "ml.batch.sample.ratio"
  val DEFAULT_ML_BATCH_SAMPLE_RATIO = 1.0
  val ML_LEARN_RATE = "ml.learn.rate"
  val DEFAULT_ML_LEARN_RATE = 0.5
  val ML_LEARN_DECAY = "ml.learn.decay"
  val DEFAULT_ML_LEARN_DECAY = 0.5
  val ML_NUM_UPDATE_PER_EPOCH = "ml.num.update.per.epoch"
  val DEFAULT_ML_NUM_UPDATE_PER_EPOCH = 10
  val ML_DECAY_INTERVALS = "ml.decay.intervals"
  val DEFAULT_ML_DECAY_INTERVALS = 50

  val ML_MINIBATCH_SIZE = "ml.minibatch.size"
  val DEFAULT_ML_MINIBATCH_SIZE = 128

  // Optimizer Params
  val DEFAULT_ML_OPTIMIZER = "Momentum"
  val ML_FCLAYER_OPTIMIZER = "ml.fclayer.optimizer"
  val DEFAULT_ML_FCLAYER_OPTIMIZER: String = DEFAULT_ML_OPTIMIZER
  val ML_EMBEDDING_OPTIMIZER = "ml.embedding.optimizer"
  val DEFAULT_ML_EMBEDDING_OPTIMIZER: String = DEFAULT_ML_OPTIMIZER
  val ML_INPUTLAYER_OPTIMIZER = "ml.inputlayer.optimizer"
  val DEFAULT_ML_INPUTLAYER_OPTIMIZER: String = DEFAULT_ML_OPTIMIZER

  val ML_FCLAYER_MATRIX_OUTPUT_FORMAT = "ml.fclayer.matrix.output.format"
  val DEFAULT_ML_FCLAYER_MATRIX_OUTPUT_FORMAT: String = classOf[RowIdColIdValueTextRowFormat].getCanonicalName
  val ML_EMBEDDING_MATRIX_OUTPUT_FORMAT = "ml.embedding.matrix.output.format"
  val DEFAULT_ML_EMBEDDING_MATRIX_OUTPUT_FORMAT: String = classOf[TextColumnFormat].getCanonicalName
  val ML_SIMPLEINPUTLAYER_MATRIX_OUTPUT_FORMAT = "ml.simpleinputlayer.matrix.output.format"
  val DEFAULT_ML_SIMPLEINPUTLAYER_MATRIX_OUTPUT_FORMAT: String = classOf[ColIdValueTextRowFormat].getCanonicalName


  // Momentum
  val ML_OPT_MOMENTUM_MOMENTUM = "ml.opt.momentum.momentum"
  val DEFAULT_ML_OPT_MOMENTUM_MOMENTUM = 0.9
  // Adam
  val ML_OPT_ADAM_GAMMA = "ml.opt.adam.gamma"
  val DEFAULT_ML_OPT_ADAM_GAMMA = 0.99
  val ML_OPT_ADAM_BETA = "ml.opt.adam.beta"
  val DEFAULT_ML_OPT_ADAM_BETA = 0.9
  // FTRL
  val ML_OPT_FTRL_ALPHA = "ml.opt.ftrl.alpha"
  val DEFAULT_ML_OPT_FTRL_ALPHA = 0.1
  val ML_OPT_FTRL_BETA = "ml.opt.ftrl.beta"
  val DEFAULT_ML_OPT_FTRL_BETA = 1.0

  // Reg param
  val ML_REG_L2 = "ml.reg.l2"
  val DEFAULT_ML_REG_L2 = 0.005
  val ML_REG_L1 = "ml.reg.l1"
  val DEFAULT_ML_REG_L1 = 0.0

  // Embedding params
  val ML_FIELD_NUM = "ml.fm.field.num"
  val DEFAULT_ML_FIELD_NUM = -1
  val ML_RANK_NUM = "ml.fm.rank"
  val DEFAULT_ML_RANK_NUM = 8

  // (MLP) Layer params
  val ML_MLP_INPUT_LAYER_PARAMS = "ml.mlp.input.layer.params"
  val DEFAULT_ML_MLP_INPUT_LAYER_PARAMS = "100,identity"
  val ML_MLP_HIDEN_LAYER_PARAMS = "ml.mlp.hidden.layer.params"
  val DEFAULT_ML_MLP_HIDEN_LAYER_PARAMS = "100,relu|100,relu|1,identity"
  val ML_MLP_LOSS_LAYER_PARAMS = "ml.mlp.loss.layer.params"
  val DEFAULT_ML_MLP_LOSS_LAYER_PARAMS = "logloss"
  val ML_NUM_CLASS = "ml.num.class"
  val DEFAULT_ML_NUM_CLASS = 2

  // MLR parameters
  val ML_MLR_RANK = "ml.mlr.rank"
  val DEFAULT_ML_MLR_RANK = 5

  // RobustRegression params
  val ML_ROBUSTREGRESSION_LOSS_DELTA = "ml.robustregression.loss.delta"
  val DEFAULT_ML_ROBUSTREGRESSION_LOSS_DELTA = 1.0

  // Kmeans params
  val KMEANS_CENTER_NUM = "ml.kmeans.center.num"
  val DEFAULT_KMEANS_CENTER_NUM = 5
  val KMEANS_SAMPLE_RATIO_PERBATCH = "ml.kmeans.sample.ratio.perbath"
  val DEFAULT_KMEANS_SAMPLE_RATIO_PERBATCH = 0.5
  val KMEANS_C = "ml.kmeans.c"
  val DEFAULT_KMEANS_C = 0.1

  // Tree Model Params
  val ML_TREE_TASK_TYPE = "ml.tree.task.type"
  val DEFAULT_ML_TREE_TASK_TYPE = "classification"
  val ML_PARALLEL_MODE = "ml.parallel.mode"
  val DEFAULT_ML_PARALLEL_MODE = "data"
  val ML_NUM_TREE = "ml.num.tree"
  val DEFAULT_ML_NUM_TREE = 10
  val ML_TREE_MAX_DEPTH = "ml.tree.max.depth"
  val DEFAULT_ML_TREE_MAX_DEPTH = 2
  val ML_TREE_MAX_NUM_NODE = "ml.tree.max.node.num"
  val ML_TREE_MAX_BIN = "ml.tree.max.bin"
  val DEFAULT_ML_TREE_MAX_BIN = 3
  val ML_TREE_SUB_SAMPLE_RATE = "ml.tree.sub.sample.rate"
  val DEFAULT_ML_TREE_SUB_SAMPLE_RATE = 1
  val ML_TREE_FEATURE_SAMPLE_STRATEGY = "ml.tree.feature.sample.strategy"
  val DEFAULT_ML_TREE_FEATURE_SAMPLE_STRATEGY = "all"
  val ML_TREE_FEATURE_SAMPLE_RATE = "ml.tree.feature.sample.rate"
  val DEFAULT_ML_TREE_FEATURE_SAMPLE_RATE = 1
  val ML_TREE_NODE_MIN_INSTANCE = "ml.tree.node.min.instance"
  val DEFAULT_ML_TREE_NODE_MIN_INSTANCE = 1
  val ML_TREE_NODE_MIN_INFOGAIN = "ml.tree.node.min.infogain"
  val DEFAULT_ML_TREE_NODE_MIN_INFOGAIN = 0
  val ML_TREE_MIN_CHILD_WEIGHT = "ml.tree.min.child.weight"
  val DEFAULT_ML_TREE_MIN_CHILD_WEIGHT = 0.01
  val ML_GBDT_REG_ALPHA = "ml.gbdt.reg.alpha"
  val DEFAULT_ML_GBDT_REG_ALPHA = 0
  val ML_GBDT_REG_LAMBDA = "ml.gbdt.reg.lambda"
  val DEFAULT_ML_GBDT_REG_LAMBDA = 1.0
  val ML_TREE_NUM_THREAD = "ml.tree.num.thread"
  val DEFAULT_ML_TREE_NUM_THREAD = 20
  val ML_GBDT_BATCH_SIZE = "ml.gbdt.batch.size"
  val DEFAULT_ML_GBDT_BATCH_SIZE = 10000
  val ML_GBDT_SERVER_SPLIT = "ml.gbdt.server.split"
  val DEFAULT_ML_GBDT_SERVER_SPLIT = false
  val ML_TREE_CATEGORICAL_FEATURE = "ml.tree.categorical.feature"
  val DEFAULT_ML_TREE_CATEGORICAL_FEATURE = ""
  val ML_TREE_IMPURITY = "ml.tree.impurity"
  val DEFAULT_ML_TREE_IMPURITY = "gini"
  val ML_TREE_AGGRE_MAX_MEMORY_MB = "ml.tree.aggr.max.memory.mb"
  val DEFAULT_ML_TREE_AGGRE_MAX_MEMORY_MB = 256

  /** AutoML **/
  val SYNC_BATCH = "ml.sync.batch"
  val DEFAULT_SYNC_BATCH = 1

  /** The loss sum of all samples */
  val TRAIN_LOSS = "train.loss"
  val VALID_LOSS = "validate.loss"
  val LOG_LIKELIHOOD = "log.likelihood"

  /** The predict error of all samples */
  val TRAIN_ERROR = "train.error"
  val VALID_ERROR = "validate.error"
}

class MLConf {}
