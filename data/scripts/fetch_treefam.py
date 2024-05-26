import glob
import os
import re
import subprocess

treefam_data_url = 'http://www.treefam.org/static/download/treefam_family_data.tar.gz'
treefam_data_zip = re.sub(r'.*/','',treefam_data_url)
if not os.path.isfile(treefam_data_zip):
    subprocess.run(['curl','-O',treefam_data_url])
    subprocess.run(['tar','xvzf',treefam_data_zip])

srcDir = treefam_data_zip.replace('.tar.gz','')
destDir = './data'
famListFile = 'families.txt'

try:
    os.mkdir(destDir)
except OSError as error:
    print(error)

startRe = re.compile('^DATA')
endRe = re.compile('^//')

srcTreeFiles = glob.glob('%s/*.nh.emf' % srcDir)
fams = [os.path.basename(f).replace('.nh.emf','') for f in srcTreeFiles]

with open (famListFile, 'w') as h:
    for fam, srcTreeFile in zip(fams,srcTreeFiles):
        print(fam)
        h.write('%s\n' % fam)
        srcAlignFile = '%s/%s.aa.fasta' % (srcDir,fam)
        destTreeFile = '%s/%s.nh' % (destDir,fam)
        destAlignFile = '%s/%s.aa.fasta' % (destDir,fam)
        with open(srcTreeFile,'r') as f:
            for line in f:
                if startRe.match(line):
                    break
            with open(destTreeFile,'w') as g:
                for line in f:
                    if endRe.match(line):
                        break
                    g.write(line)
        with open(srcAlignFile,'r') as f:
            with open(destAlignFile,'w') as g:
                for line in f:
                    if endRe.match(line):
                        break
                    g.write(line)

print('Processed %d families' % len(fams))
