# clone yolov5 repo
git clone https://github.com/ultralytics/yolov5.git

# cd to yolov5
cd ./yolov5 || exit

# install required packages
pip install -r requirements.txt

# test training
python train.py --batch-size 8 --epochs 10 --data data.yaml --name <model name> --cfg yolov5s.yaml --patience 5

# save best model into open-cv format (onnx)
python export.py --weights runs/train/<model name>/weights/best.pt --include onnx --simplify

# Remember to copy the best.onnx (portable model) to ROOT
mv runs/train/<model name>/weights/best.onnx ./..