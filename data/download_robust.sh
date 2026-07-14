#!/bin/bash
# A more robust version of the downloader
#
for idx in $(seq -f "%05g" 0 2147); do
	f=action_value-${idx}-of-02148_data.bag
	url=https://storage.googleapis.com/searchless_chess/data/train/${f}

	if [ -s "$f" ]; then
		continue
	fi

	echo "downloading $f"
	until curl -fL --retry 20 --retry-delay 5 --retry-all-errors -C - -o "$f.part" "$url"; do
		echo "retry $f"
		sleep 2
	done
	mv "$f.part" "$f"
done
