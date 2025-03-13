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

### 独自データセットの作り方

この研究室では，```CVAT```というツールを使うことが多い．

画像のラベリングでは使いやすいと思う．

https://docs.cvat.ai/docs/manual/basics/create_an_annotation_task/
