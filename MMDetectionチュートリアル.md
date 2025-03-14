# MMDetectionを使おう！

## チュートリアル

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

<details><summary>独自データセットのファイル構成</summary>

```
mmdetection---data---mydata---train
                            |
                            |-val
                            |
                            |-train_annotation.json
                            |
                            |-val-annotation.json
```
</details>

## 推論結果を使って別タスクを行うには

主に以下の2つの方法がある．
1. mmdetectionの推論コードを読んで改造する
2. mmdeployを用いてなんとかする

### mmdetectionの推論コードを読んで改造する

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
