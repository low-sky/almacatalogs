from astropy.table import Table 
import sklearn.cluster as cluster
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
t = Table.read('califa_sample.csv')
coords = SkyCoord(ra=t['RA'],dec=t['DE'],unit=(u.deg,u.deg))

km = cluster.MiniBatchKMeans(n_clusters=20 ,max_iter=100,batch_size=4)
batch = km.fit(np.c_[t['RA'].data,t['DE'].data])
#d = np.zeros((len(coords),len(coords)))
#for idx,row in enumerate(coords):
#    d[:,idx] = (coords.separation(coords[idx])).value
