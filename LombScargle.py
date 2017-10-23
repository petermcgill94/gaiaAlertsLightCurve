import numpy as np
from astroML.time_series import lomb_scargle
from astropy.io import ascii
import matplotlib.pyplot as plt
import urllib2


data = ascii.read('data/Gaia14aab.csv',format='csv')
plotting the motion of one lens.

lensRa, lensDec, id,pmra,pmdec,ref_epoch = sqlutil.get('select ra, dec, source_id,pmra,pmdec,ref_epoch from gaia_dr1.tgas_source where POWER(pmra,2) + POWER(pmdec,2) > 1000000',
                       db='wsdb',host='cappc127.ast.cam.ac.uk', user='peter_mcgill', password='Ln3g.wsk')

nullmask = data['averagemag.'] != 'null'

data = data[nullmask]

omega = np.linspace(0.01,2,1000.0)


P_LS = lomb_scargle(data['JD'],data['averagemag.'],1,omega,generalized=True)

print(P_LS)

plt.scatter(data['JD'],data['averagemag.'])
plt.show()
