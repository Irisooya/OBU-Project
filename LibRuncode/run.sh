model_path="./models/UM"
output_path="./outputs/UM"
log_path="./logs/UM"
seed=0

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}
#CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}


#--model_path="./models/UM" --log_path="./logs/UM"  --output_path="./outputs/UM" --seed 0
#PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=0
read -p "按 Enter 键退出脚本"