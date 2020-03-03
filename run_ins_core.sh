export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE

exe=$1
batchsize=$2
cpu_sum=$3
ins_sum=$4
data_size=1000
sub_data_size=$(($data_size / $ins_sum))
#echo "data_size=$sub_data_size"
cores_per_ins=$(($cpu_sum / $ins_sum))
#echo $cores_per_ins

rm -rf *.json
if [ $batchsize -lt 64 ]
then
    export LRU_CACHE_CAPACITY=1024
else
    export LRU_CACHE_CAPACITY=10
fi

echo "LRU_CACHE_CAPACITY $LRU_CACHE_CAPACITY"
for ((index=0;index<${ins_sum};index++))
do
   #echo $index
   cpus="$(($cores_per_ins * $index))-$(($cores_per_ins * ($index + 1) - 1))"
   #echo $cpus
   #export GOMP_CPU_AFFINITY="${cpus}"
   #echo $GOMP_CPU_AFFINITY

   if [ $ins_sum -ne 1 ]
   then
       if [ $index -lt $((${ins_sum} / 2)) ]
       then
           m_para=0
       else
           m_para=1
       fi
       #echo "m is :$m_para"
       memory_para="-m $m_para"
   else
       memory_para=""
   fi

   #echo "numactl -C ${cpus} ${memory_para} python $exe  --index=${index} --data-size=${sub_data_size} --batch-size=$batchsize &"

   numactl -C ${cpus} ${memory_para} python $exe  --index=${index} --data-size=${sub_data_size} --batch-size=$batchsize &
done	


