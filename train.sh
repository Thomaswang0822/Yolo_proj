# clone yolov5 repo
git clone https://github.com/ultralytics/yolov5.git

# cd to yolov5
cd ./yolov5 || exit

# install required packages
pip install -r requirements.txt

# test training
python3 train.py --batch-size 8 --epochs 10 --data data.yaml --name Model_Test --cfg yolov5s.yaml --patience 5
