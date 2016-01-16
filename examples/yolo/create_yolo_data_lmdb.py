import caffe
import lmdb
from PIL import Image
import numpy as np
import os
import shutil

target_size = (448, 448)
S = 7
classes = 20

def convert(size, box):
	dw = 1./size[0]
	dh = 1./size[1]
	x = (box[0] + box[1])/2.0
	y = (box[2] + box[3])/2.0
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return (x,y,w,h)

with open("./data/VOC/2012_train_caffe.txt") as f:
	lists = f.read().split('\n')
total = int(lists[0])
idx = 1
images = []
label_and_bbox_s = []
while idx < len(lists):
#while idx < 10:
	if lists[idx].startswith("#"):
		images.append(lists[idx+1])
		channel = int(lists[idx+2])
		width   = int(lists[idx+3])
		height  = int(lists[idx+4])
		num_box = int(lists[idx+5])
		lb = np.zeros((1, 49, 25))
		for i in range(0, num_box):
			temp_l = [ int(s) for s in lists[idx+5+1+i].split(' ')]
			class_index = temp_l[0]
			box = convert((width, height), temp_l[1:])
			x = box[0]
			y = box[1]
			col = int(x*S)
			row = int(y*S)
			x = x*S - col
			y = y*S - row
			grid_idx = col+row*S
			if lb[0, grid_idx, 0] == 1:	# already has object in this grid
				continue
			lb[0, grid_idx, 0] = 1
			#if class_index >= 0 && class_index < classes:
			lb[0, grid_idx, class_index+1] = 1
			for j in range(0, 4):
				lb[0, grid_idx, 21+j] = box[j]
			#print grid_idx, class_index, lb[0, grid_idx]
		label_and_bbox_s.append(lb)
		idx += 5+num_box
	idx+=1

if False:
	images = images[0:10]
	label_and_bbox_s = label_and_bbox_s[0:10]
	print label_and_bbox_s[0][0]

#print images
print "train set: %d images" % (len(images))

data_lmdb = './examples/yolo/yolo-train-lmdb'
label_lmdb = './examples/yolo/yolo-train-label-lmdb'

if os.path.exists(data_lmdb):
	shutil.rmtree(data_lmdb)
if os.path.exists(label_lmdb):
	shutil.rmtree(label_lmdb)

print "write image data to lmdb..."
image_db = lmdb.open(data_lmdb, map_size=int(1e12))
with image_db.begin(write=True) as txn:
	for idx, image_fn in enumerate(images):
		if idx % 500 == 0:
			print idx
		# load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
		im = np.array(Image.open(image_fn).resize((448, 448), Image.BILINEAR))
		im = im[:, :, ::-1]
		im = im.transpose((2, 0, 1))
		im_dat = caffe.io.array_to_datum(im)
		txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())
image_db.close()
print "done."

print "write image label to lmdb..."
label_db = lmdb.open(label_lmdb, map_size=int(1e12))
with label_db.begin(write=True) as txn:
	for idx, lb in enumerate(label_and_bbox_s):
		if idx % 500 == 0:
			print idx
		lb_data = caffe.io.array_to_datum(lb)
		txn.put('{:0>10d}'.format(idx), lb_data.SerializeToString())
label_db.close()
print "done."