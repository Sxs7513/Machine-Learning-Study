import cv2
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
current_palette = list(sns.xkcd_rgb.values())


def parse_anno(annotation_path):
    anno = open(annotation_path, "r")
    result = []

    for line in anno:
        s = line.strip().split(" ")
        image = cv2.imread(s[0])
        image_h, image_w = image.shape[:2]
        s = s[1:]
        box_cnt = len(s) // 5
        
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5 + 0]), float(s[i*5 + 1]), float(s[i*5 + 2]), float(s[i*5 + 3])
            width = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h
            result.append([width, height])
    
    result = np.asarray(result)
    return result


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def kmeans(boxes, k, dist=np.median,seed=1):
    # how many boxes
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)
    
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row  in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        # distribute boxes to nearest clusters based on the distance
        nearest_clusters = np.argmin(distances, axis=1)
        
        # 不再变化
        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters
    
    return clusters, nearest_clusters, distances


def plot_cluster_result(clusters, nearest_clusters, WithinClusterSumDist, wh, k):
    for icluster in np.unique(nearest_clusters):
        pick = nearest_clusters == icluster
        c = current_palette[icluster]
        plt.rc('font', size=8)
        plt.plot(wh[pick, 0], wh[pick, 1],"p",
                 color=c,
                 alpha=0.5, label="cluster = {}, N = {:6.0f}".format(icluster, np.sum(pick)))
        plt.text(clusters[icluster, 0],
                 clusters[icluster, 1],
                 "c{}".format(icluster),
                 fontsize=20, color="red")
        plt.title("Clusters=%d" %k)
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist))
    plt.tight_layout()
    plt.savefig("./kmeans.jpg")
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="voc")
    parser.add_argument("--cluster_num", type=int, default=9)
    args = parser.parse_args()

    dataset_txt = './data/%s/train.txt' % args.dataset
    anchors_txt = './data/%s_anchors.txt' % args.dataset
    anno_result = parse_anno(dataset_txt)

    clusters, nearest_clusters, distances = kmeans(anno_result, args.cluster_num)

    # sort by area
    area = clusters[:, 0] * clusters[:, 1]
    indice = np.argsort(area)
    clusters = clusters[indice]
    
    with open(anchors_txt, "w") as f:
        for i in range(args.cluster_num):
            width, height = clusters[i]
            f.writelines(str(width) + " " + str(height) + " ")

    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]), nearest_clusters])
    plot_cluster_result(clusters, nearest_clusters, 1-WithinClusterMeanDist, anno_result, args.cluster_num)
