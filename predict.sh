# recommended paddle.__version__ == 2.0.0
python tools/infer_rec.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_config.yml -o Global.pretrained_model=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest Global.infer_img=doc/imgs_words/ch/word_1.jpg
#python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml


python tools/infer/predict_rec.py --image_dir="doc/imgs_words/ch/word_1.jpg" --rec_model_dir="inference/my_model" --rec_image_shape="3, 32, 100" --rec_char_dict_path="C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\ppocr\utils\cht_tra_characters_v2.txt"

