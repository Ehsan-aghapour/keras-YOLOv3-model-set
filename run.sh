
n=1
source ../env_qe/bin/activate

for (( i=0; i<$n; i++  ))
do
   echo "Running thread $i ..."
   python eval_multiThread.py --model_path=../Quantization/Yolo_files/YoloV3_selective_half_quztized.tflite --eval_type=VOC --thread_indx=$i --num_threads=$n &
done
wait
