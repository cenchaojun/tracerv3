#TRACER
#├── data
#│   ├── custom_dataset
#│   │   ├── sample_image1.png
#│   │   ├── sample_image2.png
#      .
#      .
#      .

# For testing TRACER with pre-trained model (e.g.)
python main.py --arch 7 --img_size 640 --save_map True
python main.py --arch 6 --img_size 576 --save_map True
python main.py --arch 5 --img_size 512 --save_map True
python main.py --arch 4 --img_size 448 --save_map True
python main.py --arch 3 --img_size 384 --save_map True
python main.py --arch 2 --img_size 352 --save_map True
python main.py --arch 1 --img_size 320 --save_map True