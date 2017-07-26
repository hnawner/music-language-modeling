for file in krnfiles/*.krn
do
 mv "$file" "${file%.krn}.txt"
done
