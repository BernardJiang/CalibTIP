sh scripts/switchmax_scale.sh 1
# sh scripts/switchmax_scale.sh 0.999
sh scripts/advanced_pipeline_resnet.sh > logresnet.txt 2>logerrresnet.txt 
sh scripts/advanced_pipeline_mobilenet.sh > logmobile.txt 2>logerrmobile.txt 
