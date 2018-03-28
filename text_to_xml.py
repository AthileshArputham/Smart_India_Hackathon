import os,sys
import xml.etree.ElementTree as ET

folder = 									#actually must give path to where the detector files are stored
for filename in os.listdir(folder):
       infilename = os.path.join(folder,filename)
       if not os.path.isfile(infilename): continue
       oldbase = os.path.splitext(filename)
       newname = infilename.replace('.txt', '.xml')
       output = os.rename(infilename, newname)
	   

	   

tree=ET.parse('a.xml')

root=tree.getroot()

for child in root:
		print(child.tag)