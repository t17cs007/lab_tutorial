# Yoloを使おう

## チュートリアル

### インストール

Anacondaでインストールする．

```bash
$ conda install -n yolo python=3.12
$ conda activate yolo
$ mkdir yolo
$ cd yolo
$ conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
$ git clone https://github.com/ultralytics/ultralytics
$ cd ultralytics
$ pip install -e . -v
```

### 推論

既存モデルで推論を行うことができる．

画像は適当に用意すること．

https://docs.ultralytics.com/modes/predict/#working-with-results
