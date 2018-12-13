package com.tencent.angel.spark.ml.tree.common;

public class Global {
    public static class Conf {
        // ML Conf
        public static final String ML_TRAIN_DATA_PATH = "spark.ml.train.path";
        public static final String ML_VALID_DATA_PATH = "spark.ml.valid.path";
        public static final String ML_VALID_DATA_RATIO = "spark.ml.valid.ratio";
        public static final double DEFAULT_ML_VALID_DATA_RATIO = 0.25;
        public static final String ML_PREDICT_DATA_PATH = "spark.ml.predict.path";
        public static final String ML_NUM_CLASS = "spark.ml.class.num";
        public static final int DEFAULT_ML_NUM_CLASS = 2;
        public static final String ML_NUM_FEATURE = "spark.ml.feature.num";
        public static final String ML_FEATURE_SAMPLE_RATIO = "spark.feature.sample.ratio";
        public static final double DEFAULT_ML_FEATURE_SAMPLE_RATIO = 1.0;
        public static final String ML_LEARN_RATE = "spark.ml.learn.rate";
        public static final double DEFAULT_ML_LEARN_RATE = 1.0;
        public static final String ML_LOSS_FUNCTION = "spark.ml.loss.func";
        public static final String ML_EVAL_METRIC = "spark.ml.eval.metric";
        public static final String DEFAULT_ML_EVAL_METRIC = "";
        public static final String ML_NUM_WORKER = "spark.ml.worker.num";
        public static final String ML_NUM_THREAD = "spark.ml.thread.num";
        public static final int DEFAULT_ML_NUM_THREAD = 1;
        // GBDT Conf
        public static final String ML_GBDT_SPLIT_NUM = "spark.gbdt.split.num";
        public static final int DEFAULT_ML_GBDT_SPLIT_NUM = 100;
        public static final String ML_GBDT_TREE_NUM = "spark.gbdt.tree.num";
        public static final int DEFAULT_ML_GBDT_TREE_NUM = 20;
        public static final String ML_GBDT_MAX_DEPTH = "spark.gbdt.max.depth";
        public static final int DEFAULT_ML_GBDT_MAX_DEPTH = 6;
        public static final String ML_GBDT_MAX_NODE_NUM = "spark.gbdt.max.node.num";
        public static final String ML_GBDT_HIST_SUBTRACTION = "spark.gbdt.hist.subtraction";
        public static final boolean DEFAULT_ML_GBDT_HIST_SUBTRACTION = true;
        public static final String ML_GBDT_LIGHTER_CHILD_FIRST = "spark.gbdt.lighter.child.first";
        public static final boolean DEFAULT_ML_GBDT_LIGHTER_CHILD_FIRST = true;
        public static final String ML_GBDT_FULL_HESSIAN = "spark.gbdt.full.hessian";
        public static final boolean DEFAULT_ML_GBDT_FULL_HESSIAN = false;
        public static final String ML_GBDT_MIN_CHILD_WEIGHT = "spark.gbdt.min.child.weight";
        public static final double DEFAULT_ML_GBDT_MIN_CHILD_WEIGHT = 0.0;
        public static final String ML_GBDT_MIN_NODE_INSTANCE = "spark.gbdt.min.node.instance";
        public static final int DEFAULT_ML_GBDT_MIN_NODE_INSTANCE = 1024;
        public static final String ML_GBDT_MIN_SPLIT_GAIN = "spark.gbdt.min.split.gain";
        public static final double DEFAULT_ML_GBDT_MIN_SPLIT_GAIN = 0.0;
        public static final String ML_GBDT_REG_ALPHA = "spark.gbdt.reg.alpha";
        public static final double DEFAULT_ML_GBDT_REG_ALPHA = 0.0;
        public static final String ML_GBDT_REG_LAMBDA = "spark.gbdt.reg.lambda";
        public static final double DEFAULT_ML_GBDT_REG_LAMBDA = 1.0;
        public static final String ML_GBDT_MAX_LEAF_WEIGHT = "spark.gbdt.max.leaf.weight";
        public static final double DEFAULT_ML_GBDT_MAX_LEAF_WEIGHT = 0.0;
    }

}
