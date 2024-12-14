if [ ! -d "logs" ]; then
    mkdir logs
fi
if [ ! -d "./logs/FuXing" ]; then
    mkdir ./logs/FuXing
fi

seq_len=30
empty_ratio=0
model_name=FT-SMNet
root_path='datasets/'
data_path='FuXing.csv'
data='FuXing'
devices='cuda:0'

for pred_len in 1
do
python -u run_model.py \
  --data $data \
  --root_path $root_path \
  --data_path $data_path \
  --model_name $model_name \
  --seq_len $seq_len \
  --empty_ratio $empty_ratio \
  --devices $devices \
  --pred_len $pred_len >logs/FuXing/$model_name'_'$seq_len'_'$pred_len'_'$empty_ratio.log 
done