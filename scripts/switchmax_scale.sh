
export max_scale=${1:-"1"}
echo "copy max_scale =" $max_scale
cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.4bit.$max_scale results/mobilenet_v2_w4a8.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.js
cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.8bit.$max_scale results/mobilenet_v2_w8a8.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.js
cp modeljson/resnet50/resnet_adaquant.piano.kdp530.scaled.onnx.json.4bit.$max_scale results/resnet50_w4a8.adaquant/resnet_adaquant.piano.kdp530.scaled.onnx.json 
cp modeljson/resnet50/resnet_adaquant.piano.kdp530.scaled.onnx.json.8bit.$max_scale results/resnet50_w8a8.adaquant/resnet_adaquant.piano.kdp530.scaled.onnx.json 





