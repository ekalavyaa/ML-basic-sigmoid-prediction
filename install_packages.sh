conda install --yes --file requirements.txt
while read requirement; do conda install --yes $requirement; done < requirements.txt 
