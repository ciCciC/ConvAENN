# Anomaly Detector app for Electrocardiogram (ECG)

## Models
- [Anomaly Detector Dense AutoEncoder Neural Network](https://github.com/ciCciC/ConvAENN/blob/master/src/nnetwork/anomAeModel.py)

## Prerequisite
- install miniforge
- create virtual env
- pip install watchdog matplotlib pandas numpy fastparquet
- pip install streamlit
- **WINDOWS**
  - conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
  - pip install jupyter matplotlib numpy pandas
  - pip install tensorflow
- **MACOS**
  - pip install jupyter matplotlib numpy pandas
  - tensorflow [installer](https://developer.apple.com/metal/tensorflow-plugin/) for MPS
- run application from terminal:
  - ```streamlit run ./app.py```

# Preview
<br/>
Data page
<br/>
<img src="images/0.png" width="600" height="400">

<br/>
Train page
<br/>
<img src="images/11.png" width="600" height="400">

<br/>
Evaluate page part 1
<br/>
<img src="images/metric.png" width="600" height="400">

<br/>
Evaluate page part 2
<br/>
<img src="images/evaluate.png" width="600" height="400">