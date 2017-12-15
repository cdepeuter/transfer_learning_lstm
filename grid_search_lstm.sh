python lstm_gcp.py 256 32 5000
python lstm_gcp.py 256 64 5000
python lstm_gcp.py 256 69 5000
python lstm_gcp.py 256 128 5000
gsutil cp -r logs gs://cdp_deeplearning_tensorboard
python lstm_gcp.py 512 48 5000
python lstm_gcp.py 512 64 5000
python lstm_gcp.py 512 96 5000
python lstm_gcp.py 512 128 5000
gsutil cp -r logs gs://cdp_deeplearning_tensorboard

python lstm_gcp.py 1024 48 2500
python lstm_gcp.py 1024 96 2500
python lstm_gcp.py 2048 48 1500
python lstm_gcp.py 2048 96 1500

gsutil cp -r logs gs://cdp_deeplearning_tensorboard