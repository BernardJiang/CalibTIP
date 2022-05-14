
export max_scale=${1:-"1"}
# echo "copy max_scale =" $max_scale
mkdir -p results/mobilenet_v2.adaquant
mkdir -p results/resnet50.adaquant
cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.4bit.$max_scale results/mobilenet_v2.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.w4.onnx.json
cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.8bit.$max_scale results/mobilenet_v2.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.w8.onnx.json
cp modeljson/resnet50/resnet_adaquant.piano.kdp530.scaled.onnx.json.4bit.$max_scale results/resnet50.adaquant/resnet_adaquant.piano.kdp530.scaled.w4.onnx.json 
cp modeljson/resnet50/resnet_adaquant.piano.kdp530.scaled.onnx.json.8bit.$max_scale results/resnet50.adaquant/resnet_adaquant.piano.kdp530.scaled.w8.onnx.json 





