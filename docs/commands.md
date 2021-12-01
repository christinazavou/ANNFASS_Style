to remove a submodule:
---
- Remove the submodule entry from .git/config
```git submodule deinit -f path/to/submodule```
- Remove the submodule directory from the superproject's .git/modules directory
```rm -rf .git/modules/path/to/submodule```
- Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
```git rm -f path/to/submodule```

to push in submodule:
---
```
cd submodule
git checkout master
git add
git commit 
git push
```
- then to update parent
```
cd parent
git add submodule
git commit
git push
```


linux terminal commands
---
- list files with date modified
```ls -l```
- remove lines with text "Trans" in vim: (after vim COMMERCIALcastle.mtl)
```:g/Trans/d```
- replace text "Trans" with "tr" in all lines in vim (after vim COMMERCIALcastle.mtl)
```:%s/Trans/tr/g```
-to find regex in vim: (after vim COMMERCIALcastle.mtl)
```:%s/d\s\(.*\)/d 1.0/g```
i.e. "d followed by one space followed by any character multiple times" replaced by "d 1.0"

```:%s#/media/graphicslab/BigData/zavou/ANNFASS_DATA#/mnt/nfs/work1/kalo/maverkiou/zavou/data#g```
```:%s#/mnt/nfs/work1/kalo/maverkiou/zavou/data#/media/graphicslab/BigData/zavou/ANNFASS_DATA#g```

- to transfer data on another machine:
```
<path on swarm2> maverkiou@gypsum.cs.umass.edu:<path on gypsum>
scp <local path> maverkiou@swarm2.cs.umass.edu:<path on swarm2>
```
- to untar a file:
```
tar -xvf pycollada_unit_obj.tar.gz
or
mkdir 15buildingsrendered && tar -xvf 15buildingsrendered.tar.gz -C 15buildingsrendered
```
- to tar a directory
```tar -czvf file.tar.gz /home/vivek/data/```
- to list directory human readable sizes
```cd /mnt/nfs/work1/kalo/maverkiou/zavou && du -sh data```
- to fetch all submodules of a git repo
```git submodule update --init --recursive```
- to fetch a specific submodule of a git repo
```git submodule update <specific path to submodule>```

slurm commands
---
- to get idle nodes
```sinfo --state=idle```
- to check a job by its id or jobs of a user
```
squeue -j 7133385
squeue -u maverkiou
```
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
