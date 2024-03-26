import cv2, sys, imutils, os
import shutil
import numpy as np
import argparse
import re
from tqdm import tqdm

class Card():
	"""Store basic info for an extracted card"""
	def __init__(self, file, face, centroid):
		self.file = file
		self.face = face
		self.centroid = centroid
		self.pos = Card.centroidToName(self.centroid)
		

	@staticmethod
	def centroidToName(centroid):
		if centroid[0] < .25:
			if centroid[1] < .5:
				return"top_left"
			else:
				return "bottom_left"
		elif centroid[1] < .5:
			return "top_right"
		else:
			return "bottom_right"

OUT_DIR = ""
		


def display(in_image, title, scale=.1):
	"""
	Display an image
	in_image: image to show
	title: title of window
	scale: scale of the window
	"""
	image = cv2.resize(in_image, (0,0), fx=scale, fy=scale) 
	# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	# image = in_image
	cv2.imshow(title, image)
	cv2.setMouseCallback(title, click_event, 1.0/scale) 
	cv2.waitKey(0)

def click_event(event, x, y, flags, scale): 
  
	# checking for left mouse clicks 
	if event == cv2.EVENT_LBUTTONDOWN: 
  
		# displaying the coordinates 
		# on the Shell 
		print(x * scale, ' ', y * scale) 
  

def subimage(image, center, theta, width, height):
	"""
	Rotates OpenCV image around center with angle theta (in deg)
	then crops the image according to width and height.
	image: opencv image to extract from
	center: center of region of interest
	theta: rotation of region in degrees
	width: width of region
	height: height of region
	retuns the sub region as an image
	"""
	shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)
	if (theta < -45):
		temp = width
		width = height
		height = temp
		theta += 90
	x = int( center[0] - width/2  )
	y = int( center[1] - height/2 )
	if x < 0 or y < 0:
		temp = width
		width = height
		height = temp
		x = max(0, int( center[0] - width/2  ))
		y = max(0, int( center[1] - height/2 ))
		theta -= 90

	matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
	image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

	# print(f"center={center}, {x},{y} -> {x+int(width)},{y+int(height)} . h={height} w={width}. theta={theta}")

	image = image[ y:y+int(height), x:x+int(width) ]

	return image

def getContours(image, thresh_val):
	"""
	Get the all contours from the image after thresholding on `thresh_val`
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (15, 15), 0)
	# gray = cv2.bilateralFilter(gray,9,75,75)
	# ret, th = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)

	binary = cv2.medianBlur(gray, 3)
	ret, edged = cv2.threshold(binary,thresh_val,255,cv2.THRESH_BINARY_INV)
	# edged = cv2.Canny(edged, 3, 30)

	# edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5,5)))
	# display(edged, "edged")
	
	cnts, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
	return cnts

def exportCards(image, thresh_val, min_area_scale, max_area_scale, face, root_name="000"):
	"""
	save out postcards from the input image
	image: loaded opencv image
	thresh_val: int to threshould the image on
	min_area_scale: miminum percent (as a float in [0,1]) that a contour must take up to be big enough
	min_area_scale: maximum percent (as a float in [0,1]) that a contour must be smaller than
	face: string, either "front" or "back"
	root_name: root of the image which is used to export to a folder
	"""
	src_area = image.shape[0]*image.shape[1]
	
	cnts = getContours(image, thresh_val)

	my_dir = os.path.join(OUT_DIR, root_name)

	if not os.path.exists(my_dir):
		os.makedirs(my_dir)

	cards = []
	exports = 0
	# for each contour
	for c in cnts:
		# get bounding rectangle
		rect = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
		box = np.array(box, dtype="int")
		contour_area = cv2.contourArea(box)
		# if the contour is big enough to be a card
		if contour_area > (src_area * min_area_scale):
			norm_centroid = [rect[0][i]/ image.shape[i] for i in range(len(rect[0]))]
			image_pos = Card.centroidToName(norm_centroid)
			outfile = os.path.join(my_dir, f"{image_pos}_{face}_{thresh_val}_{exports}.png")
			# get subimage of the contour
			sub = subimage(image, rect[0], rect[2], rect[1][0], rect[1][1])
			if contour_area > (src_area * max_area_scale):
				sub_cards = exportCards(sub, thresh_val - 10, .25, .75, face, root_name=root_name)
				sub_start_corner = box[0]
				for i, sub_card in enumerate(sub_cards):
					sub_centroid = sub_card.centroid
					sub_centroid = [sub_centroid[i] * sub.shape[i] for i in range(len(sub_centroid))]
					sub_centroid = [sub_centroid[i] + sub_start_corner[i] for i in range(len(sub_centroid))]
					sub_centroid = [sub_centroid[i] / image.shape[i] for i in range(len(sub_centroid))]
					new_pos = Card.centroidToName(sub_centroid)
					new_name = sub_card.file.replace(sub_card.pos, new_pos)
					sub_card.pos = new_pos
					sub_card.centroid = sub_centroid
					shutil.move(sub_card.file, new_name)
					sub_card.file = new_name
					sub_cards[i] = sub_card
				cards += sub_cards
				continue
			sub = cv2.rotate(sub, cv2.ROTATE_90_CLOCKWISE)
			cards.append(Card(outfile, face, norm_centroid))
			cv2.imwrite(outfile, sub)
			exports += 1
			# cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
			# cv2.circle(image, tuple([int(x) for x in rect[0]]), 10, (255,0,0), 2)
	return cards


def main():
	"""
	Get the input and output folders and for all images extract the postcards 
	and save them to the output subfolders
	"""
	parser = argparse.ArgumentParser(description='Extract the cards')
	parser.add_argument('scans', type=str,
						help='folder with the scans')
	parser.add_argument('output', type=str,
						help='folder to write cards to')
	args = parser.parse_args()
	global OUT_DIR
	OUT_DIR = args.output

	front = True
	prev_num = 0
	for img in tqdm(os.listdir(args.scans)):
		num = int(re.search(r'\d+', img).group())
		num = str(num)
		img = os.path.join(args.scans, img)
		if front:
			exportCards(cv2.imread(img), 190, .15, .3, "front", root_name=num)
			exportCards(cv2.imread(img), 210, .15, .3, "front", root_name=num)
			front = False
			prev_num = num
		else:
			exportCards(cv2.imread(img), 220, .15, .3, "back", root_name=prev_num)
			front = True

if __name__ == '__main__':
	main()