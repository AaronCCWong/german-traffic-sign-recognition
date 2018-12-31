# German Traffic Sign Recognition

This repo contains models that I used for the german traffic sign classification competition done as an assignment for CS-2271 at NYU. Read about it on [Medium](https://medium.com/@aaronwong_65108/traffic-sign-classification-6e7113d9c4d5).

## Accuracy achieved using ResNet18 with data augmentation

|            | Accuracy Values                             |
|------------|---------------------------------------------|
| Training   | ![training accuracy](/assets/train_acc.png) |
| Validation | ![validation accuracy](/assets/val_acc.png) |

## Training

It is assumed that python3 is being used. Get the zipfiles for the training dataset from [here](https://www.kaggle.com/c/nyu-cv-fall-2018/data).
Put the zipfile in `data/`. Then run:

```bash
python main.py
```

## Acknowledgements

The training script is a modified version of the template provided by [@soumith](https://github.com/soumith).
