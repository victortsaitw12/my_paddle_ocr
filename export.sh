# recommended paddle.__version__ == 2.0.0
python tools/export_model.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_config.yml -o Global.pretrained_model=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest  Global.save_inference_dir=./inference/my_model/

python tools/infer/predict_rec.py --image_dir="doc/imgs_words/ch/word_1.jpg" --rec_model_dir="inference/my_model" --rec_image_shape="3, 32, 100" --rec_char_dict_path="C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\ppocr\utils\cht_tra_characters_v2.txt"


# SVTR

python tools/export_model.py -c C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\configs\rec\PP-OCRv3\my_svtr_config.yml -o Global.pretrained_model=C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\output\my_model\latest  Global.save_inference_dir=./inference/my_model/

python tools/infer/predict_rec.py --image_dir="doc/imgs_words/ch/word_1.jpg" --rec_model_dir="inference/my_model" --rec_image_shape="3, 32, 100" --rec_char_dict_path="C:\Users\victor\OneDrive\桌面\react_practice\PaddleOCR\ppocr\utils\cht_tra_characters_v2.txt"
