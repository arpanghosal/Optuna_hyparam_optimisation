#! /bin/bash


# Iterate through .cfg files in the current directory
for file in *.cfg; do
  if [ -f "$file" ]; then
    echo ">>>>>>>>>>>> Running mva trainer on $file   >>>>>>>>>>>>>>"
    job_line=$(grep "Job = " "$file")
    job_name=$(echo "$job_line" | awk -F "=" '{print $2}' | tr -d '[:space:]')
    mkdir "$job_name"
    cp "$file" "$job_name"
 
    python3 /data/atlastop3/ghosal/mva-trainer/python/mva-trainer.py -c "$file" --convert 
    python3 /data/atlastop3/ghosal/mva-trainer/python/mva-trainer.py -c "$file" --train
    python3 /data/atlastop3/ghosal/mva-trainer/python/mva-trainer.py -c "$file" --evaluate
    echo ">>>>>>>>>> Done with $file >>>>>>>"
  fi
done


 #python3 /data/atlastop3/ghosal/mva-trainer/python/mva-trainer.py -c bdt_test.cfg --optimise --optimisationpath new_configs/ --nModels 30
