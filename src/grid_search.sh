for seed in 0 1 2 3 4
do
  for k in 3 5
  do
    for relevance_scores in binary tf-idf
    do
      for lambda in .5 1 2 4 8
      do
        echo ...
        python3 train.py -r "$relevance_scores" -u y -b 32 -l 1e-5 -k "$k" --lambda "$lambda" -o model -m bert-joint -s "$seed" --gpu 1
      done
    done
  done
done
