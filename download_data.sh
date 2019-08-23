#!/bin/bash
FILE="kaggle.json"
if [ -f  "$FILE" ];
then
	mkdir ~/.kaggle
	mv kaggle.json ~/.kaggle/

	kaggle datasets download -d shadabhussain/flickr8k
	unzip -q flickr8k.zip
	unzip -q Flickr_Data.zip
	mv Flickr_Data/Images .
	mv Flickr_Data/Flickr_TextData/Flickr8k.token.txt captions.txt
	rm -r flickr8k.zip Flickr_Data.zip Flickr_Data model_weights.h5 train_encoded_images.p

	kaggle datasets download -d watts2/glove6b50dtxt
	unzip -q glove6b50dtxt.zip
	rm -r glove6b50dtxt.zip 
else
	echo "File $FILE does not exist" >&2
fi