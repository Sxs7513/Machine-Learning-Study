# solver 提供的 test 只能用于 coco 数据集，该文件提供为任意图片生成文本的能力
from core.vggnet import Vgg19
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate

import tensorflow as tf
import numpy as np
from scipy import ndimage, misc


def test(image_batch_file):
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        image_batch = np.array(
            map(
                lambda x: misc.imresize(
                    ndimage.imread(x, mode="RGB"),
                    (224, 224)
                ),
                image_batch_file
            )
        )
        feats = sess.run(vggnet.features, feed_dict={ vggnet.images: image_batch })

    data = {
        features: feats,
        file_names: image_batch_file
    }

    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    model = CaptionGenerator(
        word_to_idx, 
        dim_feature=[196, 512], 
        dim_embed=512,
        dim_hidden=1500,
        n_time_step=16, 
        prev2out=True, 
        ctx2out=True, alpha_c=1.0, 
        selector=True, dropout=True
    )

    solver = CaptioningSolver(
        model, 
        data, 
        data,  
        batch_size=128,
        test_model='./model/lstm3/model-18',
        print_bleu=False, 
        log_path='./log/'
    )

    solver.test(data)


if __name__ == '__main__':
    test(image_batch_file=['./test_images/xx'])
