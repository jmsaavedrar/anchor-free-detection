import os
import xml
import xml.etree.ElementTree as ET
import numpy as np 
import matplotlib.pyplot as plt
import PIL
import matplotlib.patches as patches


#you must downloand the cat_dog_det dataset
datadir = '/home/jsaavedr/Descargas/cat_dog_det'
imdir  = os.path.join(datadir, 'images')
anndir  = os.path.join(datadir, 'annotations')
listf = os.path.join(datadir,'train.txt')
with open(listf, 'r+') as f :
    list = [item.strip() for item in f]
hs = []
ws = []    
cats = 0
dogs = 0
for im in list :    
    tree = ET.parse(os.path.join(anndir, im + '.xml'))
    root = tree.getroot()
    objs = root.findall('object')
    imagefile = os.path.join(imdir, im + '.png')
    fig, ax = plt.subplots()
    image = PIL.Image.open(imagefile)
    ax.imshow(image)
    for obj in objs :
        cl = obj.find('name')            
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        h = ymax - ymin
        w = xmax - xmin
        hs.append(h)
        ws.append(w)
        if cl.text == 'cat' :
            cats = cats + 1
        if cl.text == 'dog' :
            dogs = dogs + 1 
        print('{} {} {} {} {}'.format(cl.text, xmin, ymin, xmax, ymax))
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.waitforbuttonpress()
    plt.show()
print('cats: {} dogs : {}'.format(cats, dogs))      
# hs = np.array(hs)
# ws = np.array(ws)
# print(hs)
# print(ws)
# hist_h, bin_edges_h = np.histogram(hs)
# hist_w, bin_edges_w = np.histogram(ws)
# fig, xa = plt.subplots(1,2)
# xa[0].bar(bin_edges_h[:-1], hist_h, width = np.diff(bin_edges_h))
# xa[1].bar(bin_edges_w[:-1], hist_w, width = np.diff(bin_edges_w))
# plt.show()
