# MMDetectionを使おう！

## チュートリアル

実行例を載せたが，どうしてもわからないときや答え合わせで読むこと．

***実行例をすぐに読んだほうが効率的に学習できるわけではない．***

英語を頑張って読む訓練を積もう．英語が苦手なら，生成系AIなどを使ってなんとか解読しよう．これも訓練である．

### インストール

以下のページを参考にインストールしよう．

https://mmdetection.readthedocs.io/en/latest/get_started.html

Anacondaインストール後は，以下のコマンドを推奨する．
```bash
$ conda create -n mmdet python=3.12
$ conda activate mmdet
$ conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
$ pip install -U openmim
$ mim install "mmengine<1.0.0"
$ mim install "mmcv>=2.0.0,<2.2.0"
$ mkdir mmdet
$ cd mmdet
$ git clone https://github.com/open-mmlab/mmdetection.git
$ cd mmdetection
$ pip install -e . -v
```

ドキュメントに従って，正しくインストールされたか確認すること．

### 推論

以下のページを参考にMMDetectionで推論すること．

ドキュメントをよく読めば問題なく実行できる．英語がんばろう．

実際に手を動かすところ．
1. [Inference with existing models で Demos の Image demo を動かすために必要なファイルのダウンロード（checkpointファイルのみ）](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html#inference-with-existing-models)
2. [Demos の Image demo を動かしてみる](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html#image-demo)

気力があれば，High-level APIs for inference - Inferencer も読んでみること．

もしmmdetectionを道具として使うなら，ここを読むと良い．

https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html

<details><summary>実行例</summary>
  
```bash
$ mkdir checkpoints
$ cd checkpoints
$ curl -LO https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
$ cd ..
$ python demo/image_demo.py demo/demo.jpg configs/rtmdet/rtmdet_l_8xb32-300e_coco.py --weights checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth --device cpu
```
</details>

### 独自データセットでの学習

以下のページを読んで，独自データセットでの学習方法を確認すること．

ドキュメントを読むべし．

https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets

<details><summary>実行例</summary>

```bash
$ cd data
$ curl -LOJ https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
$ unzip balloon_dataset.zip
$ touch convert_balloon_to_coco.py
### https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-formatにあるpythonコードをconvert_balloon_to_coco.pyにコピペする
$ python convert_balloon_to_coco.py
$ cd ..
$ mkdir balloon_config
$ cd balloon_config
$ touch mask_rcnn.py
### https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#prepare-a-configにあるpythonコードをmask_rcnn.pyにコピペする
$ cd ..
$ python tools/train.py balloon_config/mask_rcnn.py
$ python tools/test.py balloon_config/mask_rcnn.py work_dirs/mask_rcnn/epoch_12.pth
```
</details>

## 発展的内容

### ファインチューニング

以下のページを参考にすると良い．

https://mmdetection.readthedocs.io/en/latest/user_guides/index.html

```Config```や```データセット```の作り方を良く確認すること．

<details><summary>Configのいじり方</summary>

基本的には，既存のConfigファイルを拡張することで学習の設定を行う．
まず，独自Configであることを示すために，別ディレクトリを作ってしまう．
```bash
$ mkdir myconfigs
$ cd myconfigs
```

次に，自分の使いたいモデルのConfigファイルを拡張していく．
今回はmask-rcnnの拡張を例にして行う．
```python
# 拡張元
_base_ = ["../configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py"]

# 学習エポックの設定などができる
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# オプティマイザの設定ができる
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=None,
    )

# 画像のデータ拡張について設定できる
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize', scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

色々試して一番良い結果となる方法を探すこと．
</details>

<details><summary>学習させる</summary>

以下のコマンドを打てばよい．
```myconfigs/mask_rcnn.py```は適宜変えること．
```bash
$ python tools/train.py myconfigs/mask_rcnn.py
```
</details>

### 独自データセットの作り方

この研究室では，```CVAT```というツールを使うことが多い．

画像のラベリングでは使いやすいと思う．

https://docs.cvat.ai/docs/manual/basics/create_an_annotation_task/

<details><summary>独自データセットのファイル構成とそのときのConfigファイル</summary>

例えば，以下の構成でデータセットを用意したとする．

```
mmdetection---data---mydata---train-xxx.png
                            |   |---yyy.png
                            |   |---zzz.png
                            |
                            |-val-aaa.png
                            |  |--bbb.png
                            |  |--ccc.png
                            |
                            |-train_annotation.json
                            |
                            |-val-annotation.json
```

すると，Configファイルは以下のように変える必要がある．
```python
data_root = "data/mydata/"
meta_info = {
  "classes" : ("cls1", "cls2")
  "palette" : [
    (220, 20, 60), (30, 170, 230)
  ]
}

train_dataloader = dict(
  dataset=dict(
    data_root=data_root,
    meta_info=meta_info,
    ann_file="train_annotation.json"
    data_prefix=dict(img="train/")))

val_dataloader = dict(
  dataset=dict(
    data_root=data_root,
    meta_info=meta_info,
    ann_file="val_annotation.json"
    data_prefix=dict(img="val/")))
```
</details>

### モデルを選ぶには

mmdetectionではモデルの一覧が示されている．

https://github.com/open-mmlab/mmdetection?tab=readme-ov-file#overview-of-benchmark-and-model-zoo

mmdetectionはなぜか更新されなくなってきている（ような気がする）．
mmdetection以外のツールで最新のモデルが使える可能性があるため，以下のサイトを見てみても良いだろう．

https://paperswithcode.com

インスタンスセグメンテーションの場合，以下のようにすれば良い感じのものを探せる．

https://paperswithcode.com/task/instance-segmentation

### 筆者が使ったことのあるモデルと感想

* Mask-RCNN : 軽い．大きい物体検出ならこれで十分
* QueryInst : 重い．小さい物体でも検出できる．precisionは低いがrecallは高いイメージ
* Mask2Former : かなり重い．小さい物体でも検出できる．precisionは高いがrecallは低いイメージ

## 推論結果を使って別タスクを行うには

主に以下の3つの方法がある．
1. mmdetectionの推論コードを使う
2. mmdetectionの推論コードを読んで改造する
3. mmdeployを用いてなんとかする

### mmdetectionの推論コードを使う

以下のページにmmdetectionで用意されている推論コードから結果を得ることができる．

https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html#output

サーバサイドでmmdetectionを使う場合はこれで事足りるだろう．

### mmdetectionの推論コードを読んで改造する

もし，mmdetectionの推論コードでは満足できない出力であれば，自力でコードを改造する必要がある．
以下はオープンソースを自分で改変していくために必要な勘などが書いてある．
どのようにやったら上手くいきそうか予想できたら，その予想が正しそうか試してみよう．
研究活動には正解がないため，この機会に勘を鍛えよう．

<details><summary>ヒント1</summary>
  
良く読まなければならないのは，```demo/image_demo.py```や```demo/video_demo.py```のコードである．
これらのコードに注目する理由は，推論時のデモコードとして利用されているため，このコード内にヒントがあるはずだ，と考えることができるため．
</details>

<details><summary>ヒント2</summary>

コード62行目に```inference_detector```関数がある．
これが，```frame```を```model```に入力し，```result```の出力が得られていることが分かる．
関数名や引数から，この関数の戻り値について調べれば良さそうだと分かる．
</details>

<details><summary>ヒント3</summary>

```inference_detector```関数の定義を確認する．
https://github.com/open-mmlab/mmdetection/blob/main/mmdet/apis/inference.py#L122

```Returns```に```DetDataSample```クラスがあるため，これを見に行けば良さそうである．
</details>


<details><summary>ヒント4</summary>

```DetDataSample```クラスの定義を確認する．
https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/det_data_sample.py

クラスの宣言直後にメンバ変数の説明がされている．
これを読んで，どの変数が何を保存しているのか確かめよう．
</details>

<details><summary>ヒント5</summary>

例えば，マスク画像を作りたいという場合，もう一度```demo/image_demo.py```や```demo/video_demo.py```のコードを振り返る必要がある．
ヒント4で得た情報から，推論結果が保存された変数からどのようにマスクを作るのか，見ていく必要がある．
この流れから，コード63行目の```visualizer.add_sample```にヒントがありそうだと分かる．
なぜなら，frameとresultを同時に引数としているため，元画像と推論結果から何かを行っていそうだと判断できるため．
</details>

<details><summary>ヒント6</summary>

githubでは関数をクリックすると，関数の定義の候補がでてくる．
関数をクリックしてみよう．
例えば，ヒント5の実装例は以下のクラスなどがある．
https://github.com/open-mmlab/mmdetection/blob/main/mmdet/visualization/local_visualizer.py#L393

この中でヒントになりそうな変数名がある．
それは```pred```という名前がついているものである．
なぜなら，```pred```は```prediction```，つまり予測という意味の英単語であり，推論とほぼ同じ意味だからだ．
ちなみに，```gt```は```Ground Truth```であり，正解という意味で使われる英単語である．
この関数を追って，どのように改造したらよいのか検討すると良い．
この```add_sample```内のコードを改造するために，生成系AIを使うのは一手である．
生成系AIを上手に使おう．
</details>

### mmdeployを用いてなんとかする

公式ドキュメントを読んでなんとかすること．
やったことがないのでわかりません．
君が先駆者だ．

## mmdetectionのdockerを立てる

サーバサイドのプログラムを書く時，研究室の都合でサーバを変えてほしいと言われるときがある．
また，ライブラリの競合が発生して面倒になるときもごくまれにある．
そのため，サーバサイドのプログラムを書く時はdockerを立てることを勧める．

公式のdockerのコードは以下のページから確認できる．

https://github.com/open-mmlab/mmdetection/blob/main/docker/Dockerfile

ただし，このままではチュートリアルで作ったときと同じ環境が作れない．
もしチュートリアルと同じ環境を作りたかったら，以下のようにコードを改変すれば良い．
pytorchのdocker fileの値，mmengineの値，mmcvの値を変えただけである．

```
ARG PYTORCH="2.5.1"
ARG CUDA="12.4"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine<1.0.0" "mmcv>=2.0.0rc4,<2.2.0"

# Install MMDetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection
```
