kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
kill $(ps aux | grep eval.py | grep -v grep | awk '{print $2}')
kill $(ps aux | grep generate_submit.py | grep -v grep | awk '{print $2}')

