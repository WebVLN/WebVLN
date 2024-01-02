name=eval

flag="--vlnbert prevalent

      --load=Downloads/ckpt/best_val

      --test_only 0
      --train valid
      --data_dir Downloads/Data
      --setting seen

      --eval_iters -1
      --batchSize 10
      --maxAction 10"

CUDA_VISIBLE_DEVICES=1 python r2r_src/train.py $flag --name $name
