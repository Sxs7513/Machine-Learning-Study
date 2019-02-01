from __future__ import division, print_function, absolute_import
import os.path
import preprocessing_RCNN as prep
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from common import create_base_alexnet


# 除了最后一层 redesigned 外，其他层均 reuse pre-train 的模型
def create_alexnet(num_classes, restore=False):
    network = create_base_alexnet()

    # 最后一层不复用
    # 最后一层是对 fine_tune 即有人工框的样本的分类，在该例子中是三个类别（包括背景）
    # 注意这个分类指的是 select 的框是否与人工框 IOU 匹配的分类，具体看 preprocessing_RCNN 中的代码
    network = fully_connected(
        network, num_classes, activation="softmax", restore=restore)
    network = regression(
        network,
        optimizer="momentum",
        loss='categorical_crossentropy',
        learning_rate=0.001)

    return network


def fine_tune_Alexnet(network, X, Y, save_model_path, fine_tune_model_path):
    # Training
    model = tflearn.DNN(
        network,
        checkpoint_path='rcnn_model_alexnet',
        max_checkpoints=1,
        tensorboard_verbose=2,
        tensorboard_dir='output_RCNN')

    if os.path.isfile(fine_tune_model_path + '.index'):
        print("Loading the fine tuned model")
        model.load(fine_tune_model_path)
    elif os.path.isfile(save_model_path + '.index'):
        print("Loading the alexnet")
        # 复用预训练的完整分类的 alexnet
        model.load(save_model_path)
    else:
        print("No file to load, error")
        return False

    # 训练
    model.fit(
        X,
        Y,
        n_epoch=200,
        validation_set=0.1,
        shuffle=True,
        show_metric=True,
        batch_size=64,
        snapshot_step=200,
        snapshot_epoch=False,
        run_id='alexnet_rcnnflowers2')
    
    model.save(fine_tune_model_path)


if __name__ == '__main__':
    data_set = config.FINE_TUNE_DATA
    if len(os.listdir(config.FINE_TUNE_DATA)) == 0:
        print('Reading Data')
        prep.load_train_proposals(
            config.FINE_TUNE_LIST, 2, save=True, save_path=data_set)
    print("loading Data")
    X, Y = prep.load_from_npy(data_set)

    restore = False
    if os.path.isfile(config.FINE_TUNE_MODEL_PATH + '.index'):
        restore = True
        print('Continue fine-tune')

    # three class include background
    net = create_alexnet(config.FINE_TUNE_CLASS, restore=restore)
    fine_tune_Alexnet(net, X, Y, config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH)