model_path="./models/UM_eval1"
output_path="./outputs/UM_eval1"
log_path="./logs/UM_eval1"
model_file="./models/UM/model_seed_0.pkl"

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}
read -p "按 Enter 键退出脚本"