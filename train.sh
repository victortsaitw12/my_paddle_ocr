# recommended paddle.__version__ == 2.0.0
python tools/train.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_config.yml


# SVTR
python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch.yml
python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch.yml -o Global.checkpoints=output\rec\svtr_ch_all\latest

python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_NOAUG.yml


python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_LBP.yml
python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_LBP.yml  -o Global.checkpoints=output\rec\svtr_ch_all\latest

python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_SSR.yml


python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_finetue_NORMAL.yml
python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_finetue_HOG.yml
python tools/train.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch_finetue_LBP.yml

#CRNN
python tools/train.py -c configs\rec\PP-OCRv3\my_crnn.yml
python tools/train.py -c configs\rec\PP-OCRv3\my_crnn.yml -o Global.checkpoints=output\rec\mv3_none_bilstm_ctc\latest

