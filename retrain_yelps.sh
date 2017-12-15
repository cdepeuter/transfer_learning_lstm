python3 load_retrain_lstm.py sentiment 1000
python3 load_retrain_lstm.py sentiment 2000
python3 load_retrain_lstm.py sentiment 5000
python3 load_retrain_lstm.py sentiment 7500
python3 load_retrain_lstm.py sentiment 10000
python3 load_retrain_lstm.py sentiment 15000
python3 load_retrain_lstm.py sentiment 20000
python3 load_retrain_lstm.py sentiment 24000
gsutil cp -r logs gs://cdp_deeplearning_tensorboard

python3 load_retrain_lstm.py stars 1000
python3 load_retrain_lstm.py stars 2000
python3 load_retrain_lstm.py stars 5000
python3 load_retrain_lstm.py stars 7500
python3 load_retrain_lstm.py stars 10000
python3 load_retrain_lstm.py stars 15000
python3 load_retrain_lstm.py stars 20000
python3 load_retrain_lstm.py stars 24000
gsutil cp -r logs gs://cdp_deeplearning_tensorboard
