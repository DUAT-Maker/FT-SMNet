if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ ! -d "logs/FuXing_MAV" ]; then
    mkdir logs/FuXing_MAV
fi

seq_len=539
empty_ratio=0
model_name=FT-SMNet
root_path='datasets/'
data_path='FuXing_MAV.csv'
data='FuXing_MAV'
devices='cuda:0'

for pred_len in 77 154 231 385
do
python -u run_model.py \
  --data $data \
  --root_path $root_path \
  --data_path $data_path \
  --model_name $model_name \
  --seq_len $seq_len \
  --empty_ratio $empty_ratio \
  --devices $devices \
  --pred_len $pred_len >logs/FuXing_MAV/$model_name'_'$seq_len'_'$pred_len'_'$empty_ratio.log 
done
