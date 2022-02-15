for ((batch=0;batch<=14;batch++))
do
    python3 run_leakgan.py $batch 0 --if_real_data
done
