echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  "Data directory for the job is :: " $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"


#### Copy AVA Val data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ucf24/images-tars/
time python tools/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking AVA val: $now"

#### list videos and check space
echo "number of videos extracted are:: "
ls ${TARGET_DIR} | wc -l
echo "Space consumption is:: "

#### du -hs ${TARGET_DIR}
now=$(date +"%T")
echo "time after checking space : $now"