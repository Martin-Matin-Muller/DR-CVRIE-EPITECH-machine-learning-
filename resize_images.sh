find archive/chest_xray/train -type f -name "*.jpeg" -exec mogrify -strip -resize 64x64! {} +
