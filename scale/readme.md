
```bash
#!/bin/bash
curl -L -o ~/Downloads/nasa-cmaps.zip\
  https://www.kaggle.com/api/v1/datasets/download/behrad3d/nasa-cmaps

```


```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

```




```bash
conda create -n ml_env python=3.12
conda install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm tqdm jupyterlab -y
conda activate ml_env
```