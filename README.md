# tdf-service
Tourism Demand Forecasting service

A Flask based web app to perform tourism demand forecasting based on the research paper,

- Rob Law, Gang Li, Davis Fong, Xin Han (2019). Tourism Demand Forecasting: A Deep Learning Approach. Annals of Tourism Research, Vol 75, March 2019, Page 410-423

For more information on how to work with the DLM model, refer to https://github.com/tulip-lab/open-code/tree/develop/DLM

This repository is structured as follows,
- app.py -> Main Flask app file. Run it via the command 'python app.py'.
- templates (HTML files for the web service)
  - main.html
  - output.html
- static (will hold the forecasting result plot generated during the request)
  - plot.jpg
- data (contains the original data from which the model is built and trained)
  - Macau2018.csv
- path
  - src (contains all the main src files for the app and the DLM)
    - DLM.py
    - Eval.py
    - Preprocess.py
    - Setting.py
  - models (contains the model and pickle files)
    - model.h5
    - reframed.pkl
    - scaler.pkl
