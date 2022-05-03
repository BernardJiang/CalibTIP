
cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.4bit.0.999 results/mobilenet_v2_w4a8.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json
cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.8bit.0.999 results/mobilenet_v2_w8a8.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json

# cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.4bit.1 results/mobilenet_v2_w4a8.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json
# cp modeljson/mobilenetv2/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json.8bit.1 results/mobilenet_v2_w8a8.adaquant/mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json


# rm results
# ln -sf ../results.backup/results.20211221.parallel results
sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 4 8 True mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json
sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 8 8 True mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json
sh scripts/integer-programing.sh mobilenet_v2 mobilenet_v2 4 8 8 8 50 loss True

# # Uncomment to run first configuration only
# #for cfg_idx in 0
# for cfg_idx in 0 1 2 3 4 5 6 7 8 9 10 11
# do
    # TODO: run bn and bias tuning in loop on all configurations
   # sh scripts/bn_tuning.sh mobilenet_v2 mobilenet_v2 8 8 $cfg_idx
   # sh scripts/bias_tuning.sh mobilenet_v2 mobilenet_v2 8 8 $cfg_idx
# done

# rm results
# ln -sf ../results.backup/results.20211221.sequential results

# sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 4 8 True mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json --seq_adaquant
# sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 8 8 True mobilenetv2_adaquant.piano.kdp530.scaled.onnx.json --seq_adaquant
# sh scripts/integer-programing.sh mobilenet_v2 mobilenet_v2 4 8 8 8 50 loss True

# Uncomment to run first configuration only
#for cfg_idx in 0
# for cfg_idx in 0 1 2 3 4 5 6 7 8 9 10 11
# do
   # TODO: run bn and bias tuning in loop on all configurations
#    sh scripts/bn_tuning.sh mobilenet_v2 mobilenet_v2 8 8 $cfg_idx
#    sh scripts/bias_tuning.sh mobilenet_v2 mobilenet_v2 8 8 $cfg_idx
# done
