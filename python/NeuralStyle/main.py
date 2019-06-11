
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import time

def print_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))

CONTENT_IMG = 'content.jpg'
STYLE_IMG = 'style5.jpg'
OUTPUT_DIR = 'neural_style_transfer_tensorflow/'

if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)

IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

NOISE_RATIO = 0.7
BETA = 5
ALPHA = 100
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def load_vgg_model(path):
    '''
	Details of the VGG19 model:
	- 四个参数依次是 卷积核高度、卷积核宽度、输入通道数、输出通道数（卷积核个数 
	- 0 is conv1_1 (3, 3, 3, 64)
	- 1 is relu
	- 2 is conv1_2 (3, 3, 64, 64)
	- 3 is relu    
	- 4 is maxpool
	- 5 is conv2_1 (3, 3, 64, 128)
	- 6 is relu
	- 7 is conv2_2 (3, 3, 128, 128)
	- 8 is relu
	- 9 is maxpool
	- 10 is conv3_1 (3, 3, 128, 256)
	- 11 is relu
	- 12 is conv3_2 (3, 3, 256, 256)
	- 13 is relu
	- 14 is conv3_3 (3, 3, 256, 256)
	- 15 is relu
	- 16 is conv3_4 (3, 3, 256, 256)
	- 17 is relu
	- 18 is maxpool
	- 19 is conv4_1 (3, 3, 256, 512)
	- 20 is relu
	- 21 is conv4_2 (3, 3, 512, 512)
	- 22 is relu
	- 23 is conv4_3 (3, 3, 512, 512)
	- 24 is relu
	- 25 is conv4_4 (3, 3, 512, 512)
	- 26 is relu
	- 27 is maxpool
	- 28 is conv5_1 (3, 3, 512, 512)
	- 29 is relu
	- 30 is conv5_2 (3, 3, 512, 512)
	- 31 is relu
	- 32 is conv5_3 (3, 3, 512, 512)
	- 33 is relu
	- 34 is conv5_4 (3, 3, 512, 512)
	- 35 is relu
	- 36 is maxpool
	- 37 is fullyconnected (7, 7, 512, 4096)
	- 38 is relu
	- 39 is fullyconnected (1, 1, 4096, 4096)
	- 40 is relu
	- 41 is fullyconnected (1, 1, 4096, 1000)
	- 42 is softmax
	'''
    vgg = scipy.io.loadmat(path)
	vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][2][0][0]
		b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
		assert layer_name == expected_layer_name
		return W, b

    
    def _conv2d_relu(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        # 卷积层的权重和偏置都使用常量，这样反向传播就对它们无效
        # 而只会对 input 进行梯度下降
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))

        return tf.nn.relu(tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

    
    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    graph = {}
    # 唯独它是可训练的，和其他的项目有所不同
    graph["input"] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
	graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
	graph['avgpool1'] = _avgpool(graph['conv1_2'])
	graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
	graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
	graph['avgpool2'] = _avgpool(graph['conv2_2'])
	graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
	graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
	graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
	graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
	graph['avgpool3'] = _avgpool(graph['conv3_4'])
	graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
	graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
	graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
	graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
	graph['avgpool4'] = _avgpool(graph['conv4_4'])
	graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
	graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
	graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
	graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
	graph['avgpool5'] = _avgpool(graph['conv5_4'])

	return graph


def load_image(path):
	image = scipy.misc.imread(path)
	image = scipy.misc.imresize(image, (IMAGE_H, IMAGE_W))
	image = np.reshape(image, ((1, ) + image.shape))
	image = image - MEAN_VALUES
	return image


print_current_time()

config = tf.ConfigProto(allow_soft_placement = True)
#config.gpu_options.per_process_gpu_memory_fraction=0.9
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    