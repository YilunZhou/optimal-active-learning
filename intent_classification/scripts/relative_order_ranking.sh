python relative_order_ranking.py --model1 lstm --criterion1 bald --model2 cnn --criterion2 bald
python relative_order_ranking.py --model1 lstm --criterion1 bald --model2 aoe --criterion2 max-entropy
python relative_order_ranking.py --model1 lstm --criterion1 bald --model2 roberta --criterion2 bald
python relative_order_ranking.py --model1 cnn --criterion1 bald --model2 aoe --criterion2 max-entropy
python relative_order_ranking.py --model1 cnn --criterion1 bald --model2 roberta --criterion2 bald
python relative_order_ranking.py --model1 aoe --criterion1 max-entropy --model2 roberta --criterion2 bald
