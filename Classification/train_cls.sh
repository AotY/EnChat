python train_cls_dual.py -vocab_path data/reddit/vocab.pt -word_vec_size 100 -encoder_type cnn -class_num 2 -data data/reddit/askqa -save_model model/askqa -batch_size 256 -optim adam -learning_rate 0.0001 -valid_batch_size 128 -cls_arch exdouble -score_fn_type MLP -dot_flag -inner_prod_flag -padding_idx 0 -enc_layers 1 -hidden_size 256 -report_every 500 -gpuid 0 

