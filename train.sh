# recommended paddle.__version__ == 2.0.0
python tools/train.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_config.yml
#python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml