#! /bin/bash
export model=${1:-"resnet"}

pushd .  >/dev/null 2>&1
echo "step 1 :" python -m onnxsim --skip-fuse-bn $model.onnx $model.simplified.onnx 
# python -m onnxsim --skip-fuse-bn $model.onnx $model.simplified.onnx 
cp $model.onnx $model.simplified.onnx 
echo "step 2 :" python /workspace/libs/ONNX_Convertor/optimizer_scripts/pytorch2onnx.py $model.simplified.onnx $model.s1.onnx --no-bn-fusion
python /workspace/libs/ONNX_Convertor/optimizer_scripts/pytorch2onnx.py $model.simplified.onnx $model.s1.onnx --no-bn-fusion
echo "step 3 :" python /workspace/libs/ONNX_Convertor/optimizer_scripts/onnx2onnx.py $model.s1.onnx -o $model.s2.onnx --no-bn-fusion -t --add-bn
python /workspace/libs/ONNX_Convertor/optimizer_scripts/onnx2onnx.py $model.s1.onnx -o $model.s2.onnx --no-bn-fusion -t 
popd >/dev/null 2>&1
