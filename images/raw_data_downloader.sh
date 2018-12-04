#!/bin/bash

files=(
2011_09_26_drive_0001
2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_26_drive_0009
2011_09_26_drive_0011
2011_09_26_drive_0013
2011_09_26_drive_0014
2011_09_26_drive_0017
2011_09_26_drive_0018
2011_09_26_drive_0048
2011_09_26_drive_0051
2011_09_26_drive_0056
2011_09_26_drive_0057
2011_09_26_drive_0059
2011_09_26_drive_0060
2011_09_26_drive_0084
2011_09_26_drive_0091
2011_09_26_drive_0093
2011_09_26_drive_0095
2011_09_26_drive_0096
2011_09_26_drive_0104
2011_09_26_drive_0106
2011_09_26_drive_0113
2011_09_26_drive_0117
2011_09_28_drive_0001
2011_09_28_drive_0002
2011_09_29_drive_0026
2011_09_29_drive_0071)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done
