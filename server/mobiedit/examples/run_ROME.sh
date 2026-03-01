
#rotate quan
python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=../hparams/ROME_new/qwen2.5-rotate-3b-0-zo-quan --data_dir=../../DemoData/benchmark/ZsRE/ZsRE-demo.json --datatype='zsre' > ./hparams_log/test_qwen2.5_rot_zsre_zo-demo-step600_v2.log 2>&1 &

# #normal quan
# python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=../hparams/ROME_new/qwen2.5-3b-4-all --data_dir=../../DemoData/benchmark/ZsRE/ZsRE-demo.json --datatype='zsre' > ./hparams_log/test_qwen2.5_zsre_zo-demo-step600_v2.log 2>&1 &