import cv2, os
import argparse
from tqdm import tqdm
import uuid
import sqlite3


def processCards(folder, outfolder, cur):
	"""
	Foreach postcard in the input folders add it to a database
	folder: path to the root folder of extracted images
	outfolder: location to output new images
	cur: cursor to the database
	"""
	sub_folders = os.listdir(folder)
	# create output folder
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)
	# create table for postcard
	cur.execute("""CREATE TABLE IF NOT EXISTS postcards 
				(id TEXT PRIMARY KEY, front TEXT, back TEXT);""")
	# create table for faces
	cur.execute("""CREATE TABLE IF NOT EXISTS face 
				(id TEXT PRIMARY KEY, postcard TEXT, is_front INTEGER, image TEXT, 
					res_x INTEGER, res_y INTEGER);""")
	for page in tqdm(sub_folders):		
		for root, dirs, files in os.walk(os.path.join(folder, page)): 
			# all cards must have a front and a back
			assert(len(files) % 2 == 0)
			for j in range(0,len(files),2):
				pair_id = str(uuid.uuid4())
				assert(files[j][:7] == files[j+1][:7])
				# copy front image
				front_id = str(uuid.uuid4())
				front_name = f"{pair_id}-front.png"
				front_image = cv2.imread(os.path.join(root, files[j+1]))
				front_size = front_image.shape[:2]
				cv2.imwrite(os.path.join(outfolder, front_name), front_image)
				# copy back image
				back_name = f"{pair_id}-back.png"
				back_id = str(uuid.uuid4())
				back_image = cv2.imread(os.path.join(root, files[j]))
				back_size = back_image.shape[:2]
				cv2.imwrite(os.path.join(outfolder, back_name), back_image)
				# insert into the faces table
				cur.execute("""
					INSERT INTO face VALUES
						(?, ?, 1, ?, ?, ?),
						(?, ?, 0, ?, ?, ?)
				""",	(front_id, pair_id, front_name, front_size[0], front_size[1],
						back_id, pair_id, back_name, back_size[0], back_size[1]))
				# insert into the postcard table
				cur.execute("""
					INSERT INTO postcards VALUES
						(?, ?, ?)
				""", (pair_id, front_id, back_id))

def main():
	parser = argparse.ArgumentParser(description='Ingest folders of extracted cards and create a SQLite database')
	parser.add_argument('cards_folder', type=str,
						help='folder with the subfolders of cards')
	parser.add_argument('image_folder', type=str,
						help="folder to write images into")
	parser.add_argument('db', default="postcards.db", type=str,
						help="sqlite3 database file")
	args = parser.parse_args()

	con = sqlite3.connect(args.db)
	with con:
		cur = con.cursor()
		processCards(args.cards_folder, args.image_folder, cur)
	con.close()

if __name__ == '__main__':
	main()