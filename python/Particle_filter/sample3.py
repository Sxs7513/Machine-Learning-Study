# https://algorithm.joho.info/programming/python/opencv-particle-filter-py/

import cv2
import numpy as np

# 追跡対象の色範囲（Hueの値域）
def is_target(roi): 
    return (roi <= 30) | (roi >= 50)

# マスクから面積最大ブロブの中心座標を算出
def max_moment_point(mask):
    # ラベリング処理
    label = cv2.connectedComponentsWithStats(mask)
    data = np.delete(label[2], 0, 0)   # ブロブのデータ
    center = np.delete(label[3], 0, 0) # 各ブロブの中心座標
    moment = data[:,4]                 # 各ブロブの面積
    max_index = np.argmax(moment)      # 面積最大のインデックス
    return center[max_index]           # 面積最大のブロブの中心座標

# パーティクルの初期化
def initialize(img, N): 
    mask = img.copy()                  # 画像のコピー
    mask[is_target(mask) == False] = 0 # マスク画像の作成（追跡対象外の色なら画素値0）
    x, y = max_moment_point(mask)      # マスクから面積最大ブロブの中心座標を算出
    w = calc_likelihood(x, y, img)     # 尤度の算出
    ps = np.ndarray((N, 3), dtype=np.float32) # パーティクル格納用の配列を生成
    ps[:] = [x, y, w]                  # パーティクル用配列に中心座標と尤度をセット
    return ps

# 1.リサンプリング(前状態の重みに応じてパーティクルを再選定)
def resampling(ps): 
    # 累積重みの計算
    ws = ps[:, 2].cumsum()
    last_w = ws[ws.shape[0] - 1]
    # 新しいパーティクル用の空配列を生成
    new_ps = np.empty(ps.shape)
    # 前状態の重みに応じてパーティクルをリサンプリング（重みは1.0）
    for i in range(ps.shape[0]):
        w = np.random.rand() * last_w
        new_ps[i] = ps[(ws > w).argmax()]
        new_ps[i, 2] = 1.0

    return new_ps

# 2.推定（パーティクルの位置）
def predict_position(ps, var=13.0): 
    # 分散に従ってランダムに少し位置をずらす
    ps[:, 0] += np.random.randn((ps.shape[0])) * var
    ps[:, 1] += np.random.randn((ps.shape[0])) * var

# 尤度の算出
def calc_likelihood(x, y, img, w=30, h=30):
    # 画像から座標(x,y)を中心とする幅w, 高さhの矩形領域の全画素を取得
    x1, y1 = max(0, x-w/2), max(0, y-h/2)
    x2, y2 = min(img.shape[1], x+w/2), min(img.shape[0], y+h/2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    roi = img[y1:y2, x1:x2]
    # 矩形領域中に含まれる追跡対象(色)の存在率を尤度として計算
    count = roi[is_target(roi)].size
    return (float(count) / img.size) if count > 0 else 0.0001

# パーティクルの重み付け
def calc_weight(ps, img): 
    # 尤度に従ってパーティクルの重み付け
    for i in range(ps.shape[0]): 
        ps[i][2] = calc_likelihood(ps[i, 0], ps[i, 1], img)

    # 重みの正規化
    ps[:, 2] *= ps.shape[0] / ps[:, 2].sum()

# 3.観測（全パーティクルの重み付き平均を取得）
def observer(ps, img):
    # パーティクルの重み付け
    calc_weight(ps, img)   
    # 重み和の計算
    x = (ps[:, 0] * ps[:, 2]).sum()
    y = (ps[:, 1] * ps[:, 2]).sum()
    # 重み付き平均を返す
    return (x, y) / ps[:, 2].sum()

# パーティクルフィルタ
def particle_filter(ps, img, N=300):
    # パーティクルが無い場合
    if ps is None:
        ps = initialize(img, N) # パーティクルを初期化
    
    ps = resampling(ps)    # 1.リサンプリング
    predict_position(ps)   # 2.推定
    x, y = observer(ps, img) # 3.観測
    return ps, int(x), int(y)

def main():
    # パーティクル格納用の変数
    ps = None

    # カメラのキャプチャ
    cap = cv2.VideoCapture('./video1.webm')

    while cv2.waitKey(30) < 0:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        h = hsv[:, :, 0]

        # S, Vを2値化（大津の手法）
        ret, s = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        ret, v = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        h[(s == 0) | (v == 0)] = 100  

        # パーティクルフィルタ
        ps, x, y = particle_filter(ps, h, 300)

        if ps is None:
            continue
        # 画像の範囲内にあるパーティクルのみ取り出し
        ps1 = ps[(ps[:, 0] >= 0) & (ps[:, 0] < frame.shape[1]) &
                          (ps[:, 1] >= 0) & (ps[:, 1] < frame.shape[0])]
        # パーティクルを赤色で塗りつぶす
        for i in range(ps1.shape[0]): 
            frame[int(ps1[i, 1]), int(ps1[i, 0])] = [0, 0, 200]     
            # パーティクルの集中部分を赤い矩形で囲む
        cv2.rectangle(frame, (x-20, y-20), (x+20, y+20), (0, 0, 200), 5)

        cv2.imshow('Result', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        