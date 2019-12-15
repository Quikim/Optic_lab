# Optic_lab
A set of tools made using matrix transfert method for ellipsometry and spectrophotometry. It uses Berreman's formalism to be able to model optical response of isotropic and anisotropic layers.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Quikim/Optic_lab/master)

# REQUIREMENTS
-Numpy

# EXAMPLE
```
import T_R_elli as tre
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#defining the wavelength of interest
lam,E=tre.wavelength(300e-9,1000e-9,200)

#Setting the SiO2 dispersion law as a Lorentz oscillator of F=12eV Gamma=0eV and En0=12eV
nkSiO2=tre.Lorentz(12,0,12,E)

tre.clearlayer() #remove any layer to avoid append this on top 
tre.my_layer(nkSiO2,100) #Adding a layer of SiO2 of 100 nm 

#Setting the SiO2 dispersion law as a Tauc-Lorentz oscillator of epsinf=1.15 F=122eV C=2.54eV, En0=3.45eV, Eg=1.12
nkSi=tre.TLorentz(1.15,122,2.54,3.45,1.2,E)

#Setting the substrate to be 1mm of Si and the incidence angle of 65°
tre.my_sub(nkSi,1e-3,65*np.pi/180)

#Plotting the result
plt.plot(lam*1e9,tre.psidel()['psi']*180/np.pi,lam*1e9,tre.psidel()['delta']*180/np.pi)
plt.ylabel('Ψ,Δ (°)')
plt.xlabel('Wavelength(nm)')
plt.show()
```


Other examples are available in the form of Jupyter notebooks. An example of Jupyter notebook used to model 1800nm SiO2 on Si (ellipsometry) and Glass (T and R) :
![Alt text](https://raw.githubusercontent.com/Quikim/Optic_lab/master/snapshot.png)
