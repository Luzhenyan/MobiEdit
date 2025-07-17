# # python run_knowedit_llama2.py --editing_method=ROME --hparams_dir=../hparams/ROME/qwen2.5-3b --data_dir=/home/pcllzy/KnowEdit/agenda/agenda.json --datatype='zsre'
python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-0 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v10_lr5e-5.log 2>&1 &
python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-1 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v10_lr1e-4.log 2>&1 &
python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-2 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v10_lr5e-4.log 2>&1 &
python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-3 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v10_lr1e-3.log 2>&1 &
python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-4 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v10_lr1e-2.log 2>&1 &
# python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-0 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v5_lr5e-2.log 2>&1 &
# python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-0 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v5_lr5e-2.log 2>&1 &
# python run_wise_editing.py --editing_method=WISE --hparams_dir=../hparams/WISE/qwen2.5-3b-0 --data_dir=/home/pcllzy/KnowEdit/benchmark --data_type=ZsRE > ./hparams_log/test_wise_v5_lr5e-2.log 2>&1 &
