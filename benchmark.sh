# recommended paddle.__version__ == 2.0.0
python tools/eval.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_config.yml -o Global.checkpoints=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest


# SVTR
python tools/eval.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_svtr_config.yml -o Global.checkpoints=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest

python tools/eval.py -c configs\rec\PP-OCRv3\my_svtrtiny_ch.yml -o Global.checkpoints=output\rec\svtr_ch_all\best_accuracy
