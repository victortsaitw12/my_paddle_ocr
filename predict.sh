# recommended paddle.__version__ == 2.0.0
python tools/infer_rec.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_config.yml -o Global.pretrained_model=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest Global.infer_img=doc/imgs_words/ch/word_1.jpg

# SVTR
python tools/infer_rec.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_svtr_config.yml -o Global.pretrained_model=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest Global.infer_img=doc/imgs_words/ch/word_1.jpg

