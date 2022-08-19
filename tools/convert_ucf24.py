
import os
import numpy as np
import json
import cv2
import pickle

OUT_PATH = '/cluster/work/cvl/gusingh/data/tracking/datasets/ucf24/'
SPLITS = ['train', 'val']
HALF_VIDEO = False
CREATE_SPLITTED_ANN = False
CREATE_SPLITTED_DET = False
SAVE_JSON = True



def main():

	anno_file  = os.path.join(OUT_PATH, 'annotations', 'pyannot_with_class_names.pkl')
	print(anno_file)
	with open(anno_file,'rb') as fff:
		final_annots = pickle.load(fff)

	database = final_annots['db']
	trainvideos = final_annots['trainvideos']
	ucf_classes = final_annots['classes']
	
	
	for split in SPLITS:
		seq_map_file = open(os.path.join(OUT_PATH, 'seqmaps', split+ '.txt'), 'w')
		seq_map_file.write('name\n')
		out_path = OUT_PATH + '{}_conf0.json'.format(split)
		categories = {}

		out = {'images': [], 'annotations': [], 
				'categories': [{'id': c+1, 'name': cname} for c, cname in enumerate(ucf_classes)],
				'videos': []}

		image_cnt = 0
		ann_cnt = 0
		video_cnt = 0

		for seq in database.keys():
			istrain_seq = seq not in trainvideos
			if split == 'train' and istrain_seq:
				continue
			elif split == 'val' and not istrain_seq:
				continue
			
			seq_map_file.write(f'{seq:s}\n')

			video_cnt += 1
			out['videos'].append({
			'id': video_cnt,
			'file_name': seq})
			num_images = int(database[seq]['numf'])

			for i in range(num_images):
				h, w = 240, 320
				image_info = {'file_name': '{}/img_{:05d}.jpg'.format(seq, i),
								'id': image_cnt + i + 1,
								'frame_id': i + 1,
								'prev_image_id': image_cnt + i if i > 0 else -1,
								'next_image_id': \
								image_cnt + i + 2 if i < num_images - 1 else -1,
								'video_id': video_cnt}

				image_info['height'] = h
				image_info['width'] = w

				out['images'].append(image_info)

			# print('{}: {} images'.format(seq, num_images)) # in case of verbose

			ann_path = os.path.join(OUT_PATH, 'annotations', seq, 'gt')
			if not os.path.isdir(ann_path):
				os.makedirs(ann_path)
			
			ann_file = open(os.path.join(ann_path,'gt.txt'),'w')

			annotations = database[seq]['annotations']
			for track_id, tube in enumerate(annotations):
				# print('tube start end fs', tube['sf'], tube['ef'])
				frames = [int(f) for f in np.arange(tube['sf'], tube['ef'], 1).astype(np.int16)]
				for frame_index, frame_id in enumerate(frames): # start of the tube to end frame of the tube
					category_id = tube['label'] + 1
					# print(category_id, frame_id)
					# assert action_id == label, 'Tube label and video label should be same'
					box = tube['boxes'][frame_index, :].copy().astype(np.int16) # get the box as an array
					ann_cnt += 1
					ann = {'id': ann_cnt,
							'category_id': category_id,
							'image_id': image_cnt + frame_id+1,
							'instance_id': track_id,
							'bbox': box.tolist(),
							'conf': 1.0,
							'iscrowd': 0
						   }
					
					ann['area'] = int(ann['bbox'][2] * ann['bbox'][3])
					out['annotations'].append(ann)

					ann_file.write('{:d},{:d},{:0.1f},{:0.1f},{:0.1f},{:0.1f},{:d},{:d},{:d}\n'.format(frame_id+1, track_id, box[0],box[1],box[2],box[3], 1, category_id, 1))

			image_cnt += num_images
	
		out_path = os.path.join(OUT_PATH, 'annotations',split+ '.json')
		print('loaded {} for {} images and {} samples'.format(
		split, len(out['images']), len(out['annotations'])))
		json.dump(out, open(out_path, 'w'))


if __name__ == '__main__':
	main()