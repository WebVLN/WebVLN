name=train

flag="--vlnbert prevalent

      --test_only 0
      --train train
      --data_dir Downloads/Data
      --setting seen

      --log_every 10000
      --eval_iters -1
      --batchSize 4
      --maxAction 10
      --feedback mix
      --lr 1e-5
      --iters 200000

      --teacherWeight 1

      --featdropout 0.4
      --dropout 0.4"

CUDA_VISIBLE_DEVICES=1 python r2r_src/train.py $flag --name $name
