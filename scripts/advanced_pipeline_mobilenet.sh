# rm results
# ln -sf ../results.backup/results.20211221.parallel results
sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 4 4 True
sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 8 8 True
# sh scripts/integer-programing.sh mobilenet_v2 mobilenet_v2 4 4 8 8 50 loss True

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

# sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 8 8 True --seq_adaquant
# sh scripts/adaquant.sh mobilenet_v2 mobilenet_v2 4 4 True --seq_adaquant
# sh scripts/integer-programing.sh mobilenet_v2 mobilenet_v2 4 4 8 8 50 loss True

# Uncomment to run first configuration only
#for cfg_idx in 0
# for cfg_idx in 0 1 2 3 4 5 6 7 8 9 10 11
# do
   # TODO: run bn and bias tuning in loop on all configurations
#   sh scripts/bn_tuning.sh mobilenet_v2 mobilenet_v2 8 8 $cfg_idx
#   sh scripts/bias_tuning.sh mobilenet_v2 mobilenet_v2 8 8 $cfg_idx
# done
