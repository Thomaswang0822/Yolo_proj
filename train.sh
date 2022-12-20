git clone https://github.com/ultralytics/yolov5.git

cd ./yolov5 || exit

pip install -r requirements.txt

python3 train.py --batch-size 8 --epochs 10 --data data.yaml --name Model_Test --cfg yolov5s.yaml --patience 5
