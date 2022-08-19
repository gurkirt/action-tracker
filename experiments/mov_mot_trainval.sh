

echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/
mkdir -p $TARGET_DIR
echo  "Data directory for the job is :: " $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"

# SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ucf24/images-tars/
# time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16

tar -xf /cluster/work/cvl/gusingh/data/tracking/mot_trainval.tar -C $TARGET_DIR
tar -xf /cluster/work/cvl/gusingh/data/tracking/ch_train.tar -C $TARGET_DIR
tar -xf /cluster/work/cvl/gusingh/data/tracking/ch_val.tar -C $TARGET_DIR

now=$(date +"%T")
echo "time after unpacking MOT TRAINVAL val: $now"




