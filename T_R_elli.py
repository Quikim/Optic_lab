import math
import numpy as np
global my_layers
my_layers=[]
hb=6.582119514*10**-16#planck constant
c=299792458 #speed of light

#Function to convert complex permittivity to refractive index
def n(eps):
    reps=np.real(eps)
    imeps=np.imag(eps)
    n=np.sqrt((np.sqrt(reps**2+imeps**2)+reps)/2)
    return n

#Function to convert complex permittivity to extinction coefficient
def k(eps):
    reps=np.real(eps)
    imeps=np.imag(eps)
    k=np.sqrt((np.sqrt(reps**2+imeps**2)-reps)/2)
    return k

#Wavelength function sets the wavelength to be analyzed for all other functions, unit is in m
def wavelength(lam_min,lam_max,number_of_point,ni=1):
    global lam,omega,E,nb,n_i
    nb=number_of_point
    lam=np.linspace(lam_min,lam_max,nb)
    omega=np.asarray(2*c*math.pi/lam) #Setting angular frequency array
    E=omega*hb#Setting energy array
    n_i=np.linspace(ni,ni,nb)
    return lam,E

#Lorentz oscillator to model permittivity dispersion law of a dielectric material. All parameters are in eV
def Lorentz(F,gamma,E0,E):
    eps=1+(F**2/(E0**2-E**2+1j*gamma*E))
    return eps

#Tauc-Lorentz oscillator to model permittivity dispersion law  of a dielectric/semiconductor close to its bandgap. All parameters are in eV
def TLorentz(epsinf,F,C,E0,Eg,E):
    ieps=(F*E0*C*(E-Eg)**2)/(((E**2-E0**2)**2+C**2*E**2)*E)*np.heaviside(E-Eg,0)
    alpha=np.sqrt(4*E0**2-C**2)
    gamma=np.sqrt(E0**2-C**2/2)
    aln=(Eg**2-E0**2)*E**2+(Eg**2*C**2)-E0**2*(E0**2+3*Eg**2)
    atan=(E**2-E0**2)*(E0**2+Eg**2)+Eg**2*C**2
    xsi4=(E**2-gamma**2)**2+alpha**2*C**2/4
    reps=epsinf+(F*C*aln/(np.pi*xsi4*2*alpha*E0))*np.log((E0**2+Eg**2+alpha*Eg)/(E0**2+Eg**2-alpha*Eg))-((F*atan/(np.pi*xsi4*E0))*(np.pi-np.arctan((2*Eg+alpha)/C)+np.arctan((-2*Eg+alpha)/C)))+((2*F*E0/(np.pi*xsi4*alpha)*Eg*(E**2-gamma**2)*(np.pi+(2*np.arctan(2*(gamma**2-Eg**2)/(alpha*C))))))-((F*E0*C*(E**2+Eg**2))/(np.pi*xsi4*E)*np.log(np.absolute(E-Eg)/(E+Eg)))+((2*F*E0*C*Eg)/(np.pi*xsi4)*np.log((np.absolute(E-Eg)*(E+Eg))/np.sqrt((E0**2+Eg**2)**2+Eg**2*C**2)))
    eps=reps+1j*ieps
    return eps

#Function to calculate the anisotropy of permettiviy matrix 
def anisotrope(eps,phi_E=math.radians(0),theta_E=math.radians(0),psi_E=math.radians(0)):
    #----Definition of the Euler angles----#
    c1=np.cos(phi_E)
    c2=np.cos(theta_E)
    c3=np.cos(psi_E)
    s1=np.sin(phi_E)
    s2=np.sin(theta_E)
    s3=np.sin(psi_E)
    Euler=np.matrix([[c1*c3-(c2*s1*s3),-c1*s3-(c2*c3*s1),s1*s2],[c3*s1+(c1*c2*s3),c1*c2*c3-(s1*s3),-c1*s2],[s2*s3, c3*s2,c2]])
    def epsp_f(eps):
        return np.dot(np.dot(Euler,eps),Euler.transpose())
    epsp=np.array(list(map(epsp_f,eps)))
    return epsp

#Bruggeman model that takes permittivity epsm, epsp as matrix material and inclusion material with a volumic fraction f as inputs and gives the effective permittivity 3x3 matrix as output
#By default the Bruggeman is set in an isotropic configuration but can also be use for anisotropic inclusion
# The parameters dep_c represents the value of the depolarizing factor Lc. La and Lb are calculated from the depolarizing factor Lc and from dep_ab_split: for a split value of 0.5 La and Lc are equals
def Bruggeman(epsm,epsp,f,dep_ab_split=0.5,dep_c=1/3):
    La=dep_ab_split*(1-dep_c)
    Lb=(1-dep_ab_split)*(1-dep_c)
    Lc=dep_c
    
    def epsa_f(epsm,epsp):
        return ((f-1+La)*epsm+(La-f)*epsp-np.sqrt(-4*(La-1)*La*epsm*epsp+((1-f-La)*epsm+(f-La)*epsp)**2))/(2*La-2)
    epsa=np.array(list(map(epsa_f,epsm,epsp)))

    def epsb_f(epsm,epsp):
        return ((f-1+Lb)*epsm+(Lb-f)*epsp-np.sqrt(-4*(Lb-1)*Lb*epsm*epsp+((1-f-Lb)*epsm+(f-Lb)*epsp)**2))/(2*Lb-2)
    epsb=np.array(list(map(epsb_f,epsm,epsp)))

    def epsc_f(epsm,epsp):
        return ((f-1+Lc)*epsm+(Lc-f)*epsp-np.sqrt(-4*(Lc-1)*Lc*epsm*epsp+((1-f-Lc)*epsm+(f-Lc)*epsp)**2))/(2*Lc-2)
    epsc=np.array(list(map(epsc_f,epsm,epsp)))
    
    eps=[np.matrix([[epsa[i],0,0],[0, epsb[i],0],[0, 0,epsc[i]]]) for i in range(nb)]
    return eps

#Definition of the class layer: permettivity, thickness (in nm), and boolean anisotrope (True for anisotrpic layer and False for isotropic)
class layer:
    def __init__(self,eps,d,anis):
        self.eps=eps
        self.d=d
        self.anis=anis
        
#Definition of the class substrate: permettivity, thickness (in m), incidence angle of the light beam in radian
class substrate:
    def __init__(self,Nsub,d,theta_i):
        self.Nsub=Nsub
        self.d=d
        self.theta_i=theta_i

#Definition of the global variable layer. This function will stack every new layer on top of existing ones
def my_layer(eps,d,anis=False):
    global my_layers
    my_layers.append(layer(eps,d,anis))
    return

#Function to clear all layer already set
def clearlayer():
    global my_layers
    my_layers=[]
    return

#Function to define substrate and incidence angle of the light beam value
def my_sub(Nsub,d,theta_i):
    global my_substrate
    my_substrate=substrate(Nsub,d,theta_i)
    return

#Function to calcultate the partial transfert matrix
def Tp(layer):
    d=layer.d*10**-9#convert input thickness from nm to m
    eps=layer.eps
    theta_i=my_substrate.theta_i
    Kxx=n_i*np.sin(theta_i)
    if(not layer.anis):
        # if layer is istoropic the eigen value can be express as :
        def q_f(eps,Kxx):
            return np.sqrt(eps-Kxx**2)
        q=np.array(list(map(q_f,eps,Kxx)))
        def Tp_f(omega,q,eps):
            a0=np.cos(omega/c*d*q)
            temp=np.sin(omega/c*d*q)
            d0=temp*q*1j/eps
            c1=temp*(-1j)/q
            b2=temp*q*(-1j)
            a3=temp*1j*eps/q
            Tp=np.matrix([[a0,0,0,d0],
                        [0,a0,c1,0],
                        [0,b2,a0,0],
                        [a3,0,0,a0]])
            return Tp
        Tp=np.array(list(map(Tp_f,omega,q,eps)))
        return Tp
    else: # if layer is anistoropic 
    #Calculus of the delta matrix#
        def delta_f(eps,Kxx):
            a0=-Kxx*(eps[2,0]/eps[2,2])
            b0=-Kxx*(eps[2,1]/eps[2,2])
            d0=1-(Kxx*Kxx/eps[2,2])
            a2=(eps[1,2]*eps[2,0]/eps[2,2])-eps[1,0]
            b2=Kxx*Kxx-eps[1,1]+(eps[1,2]*eps[2,1]/eps[2,2])
            d2=(Kxx*eps[1,2]/eps[2,2])
            a3=(-eps[0,2]*eps[2,0]/eps[2,2])+eps[0,0]
            b3=-(eps[0,2]*eps[2,1]/eps[2,2])+eps[0,1]
            d3=(-Kxx*eps[0,2]/eps[2,2])
            delta=np.matrix([[a0,b0,0,d0],[0,0,-1,0],[a2,b2,0,d2],[a3,b3,0,d3]])
            return delta
        delta=np.array(list(map(delta_f,eps,Kxx)))
        
        #Calculus of the eigenvalues matrix#
        def q_f(delta):
            return np.linalg.eig(delta)[0]
        q=np.array(list(map(q_f,delta)))

        def beta_f(q,omega):
            a1=np.exp(-d*1j*q[0]*omega/c)/((q[0]-q[1])*(q[0]-q[2])*(q[0]-q[3]))
            b1=np.exp(-d*1j*q[1]*omega/c)/((q[1]-q[2])*(q[1]-q[3])*(q[1]-q[0]))
            c1=np.exp(-d*1j*q[2]*omega/c)/((q[2]-q[3])*(q[2]-q[0])*(q[2]-q[1]))
            d1=np.exp(-d*1j*q[3]*omega/c)/((q[3]-q[0])*(q[3]-q[1])*(q[3]-q[2]))
            beta_0=-q[1]*q[2]*q[3]*a1-q[2]*q[3]*q[0]*b1-q[3]*q[0]*q[1]*c1-q[0]*q[1]*q[2]*d1
            beta_1=((q[1]*q[2])+(q[1]*q[3])+(q[2]*q[3]))*a1+((q[2]*q[3])+(q[2]*q[0])+(q[3]*q[0]))*b1+((q[3]*q[0])+(q[3]*q[1])+(q[0]*q[1]))*c1+((q[0]*q[1])+(q[0]*q[2])+(q[1]*q[2]))*d1
            beta_2=-(q[1]+q[2]+q[3])*a1-(q[2]+q[3]+q[0])*b1-(q[3]+q[0]+q[1])*c1-(q[0]+q[1]+q[2])*d1
            beta_3=a1+b1+c1+d1
            return [beta_0,beta_1,beta_2,beta_3]
        beta=np.array(list(map(beta_f,q,omega)))
        def Tp_f (beta,delta):
            return beta[0]*np.identity(4)+beta[1]*delta+beta[2]*np.dot(delta,delta)+beta[3]*np.dot(delta,np.dot(delta,delta))
        Tp=np.array(list(map(Tp_f,beta,delta)))
    return Tp

#Function to calcultate the global transfert matrix at the outside medium/layer/substrate interface
def t():
    fmatrix=[]
    theta_i=my_substrate.theta_i
    nTp=[Tp(my_layers[j]) for j in range(len(my_layers))]
    nt=my_substrate.Nsub
    def costheta_t_f (n_i,nt):
        return np.sqrt(1-(n_i*np.sin(theta_i)**2/(nt**2)))
    costheta_t=np.array(list(map(costheta_t_f,n_i,nt)))
    
    def Lt_f (costheta_t,nt):
        return np.matrix([[0,0,costheta_t,0],[1,0,0,0],[-nt*costheta_t,0,0,0],[0,0,nt,0]])
    Lt=np.array(list(map(Lt_f,costheta_t,nt)))

    def Li_f(n_i):
        return 1/2*np.matrix([[0,1,-1/(n_i*np.cos(theta_i)),0],[0,1,1/(n_i*np.cos(theta_i)),0],[1/np.cos(theta_i),0,0,1/n_i],[-1/np.cos(theta_i),0,0,1/n_i]])
    Li=np.array(list(map(Li_f,n_i)))
    
    fmatrix.append(Li)
    for j in range(len(nTp)):
        fmatrix.append(nTp[j])
    fmatrix.append(Lt)
    T=[np.linalg.multi_dot([(fmatrix[j][i]) for j in range(len(fmatrix))])for i in range(nb)]
    return T

#Function to calcultate the transmitted and relflected Jones matrix from the global transfert matrix
def Jt_Jr():
    T=t()
    def Jt_Jr_f(T):
        a=T[0,0]*T[2,2]-T[0,2]*T[2,0]
        #Calculus of the Jones transmittance coefficients at the incident medium Layer+substrate interface#
        Tss=T[2,2]/a
        Tsp=-T[2,0]/a 
        Tps=-T[0,2]/a
        Tpp=T[0,0]/a
        #Calculus of the Jones reflectance coefficients at the incident medium Layer+substrate interface#
        Rpp=(T[0,0]*T[3,2]-T[0,2]*T[3,0])/a
        Rsp=(T[0,0]*T[1,2]-T[0,2]*T[1,0])/a
        Rss=(T[1,0]*T[2,2]-T[1,2]*T[2,0])/a
        Rps=(T[2,2]*T[3,0]-T[2,0]*T[3,2])/a
        Jr=np.matrix([[Rpp,Rps],[Rsp,Rss]])
        Jt=np.matrix([[Tpp,Tps],[Tsp,Tss]])
        return {'Jr':Jr,'Jt':Jt}
    JtJr=np.array(list(map(Jt_Jr_f,T)))
    Jt=[JtJr[i]['Jt'] for  i in range(nb)]
    Jr=[JtJr[i]['Jr'] for  i in range(nb)]
    return {'Jr':Jr,'Jt':Jt}

#function to calculate the reflected Muller matrix from Jones matrix
def Mjr():
    J=Jt_Jr()
    nJr=J['Jr']
    #Calculus of the reflected Mueller matrice from Jones matrix#
    A=[[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1j,-1j,0]]
    def Mjr_f(nJr):
        return np.dot(A,np.dot(np.kron(nJr,np.conjugate(nJr)),np.linalg.inv(A)))
    Mjr=np.array(list(map(Mjr_f,nJr)))
    return Mjr

#function to calculate the transmitted Muller matrix from Jones matrix
def Mjt():
    #Calculus of the transmited Mueller matrices from Jones matrix#
    J=Jt_Jr()
    nJt=J['Jt']
    A=[[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1j,-1j,0]]
    def Mjt_f(nJr):
        return np.dot(A,np.dot(np.kron(nJt,np.conjugate(nJt)),np.linalg.inv(A)))
    Mjt=np.array(list(map(Mjt_f,nJt)))
    return Mjt

#function to calculate the normalized reflected Muller matrix from muller matrix
def Mnjr():
    #Calculus of the normalized Mueller matrices #
    nMjr=Mjr()
    Mnjr=[nMjr[i]/nMjr[i][0,0] for i in range(nb)]
    return Mnjr

#function to calculate the normalized transmitted Muller matrix from muller matrix
def Mnjt():
    nMjt=Mjt()
    Mnjt=[nMjt[i]/nMjt[i][0,0] for i in range(nb)]
    return Mnjt

#function to calculte the ellipsometric angle psi and delta from the reflected jones matrix
def psidel():
    nJ=Jt_Jr()
    J=nJ['Jr']
    prp=[np.absolute(J[i][0,0]) for i in range(nb)]
    prs=[np.absolute(J[i][1,1]) for i in range(nb)]
    drp= [J[i][0,0] for i in range(nb)]
    drs= [J[i][1,1] for i in range(nb)]
    delta=np.asarray([np.angle(drp[i]/drs[i], deg=False) for i in range(nb)])
    psi=np.arctan2(prp, prs)
    return {'psi':np.tan(psi),'delta':-np.cos(delta)}

#function to calculte the partial transfert matrix contribution from the beam reflected on the backside of the substrate 
def Tpb(layer):
    d=layer.d*10**-9#convert input thickness from nm to m
    eps=layer.eps
    theta_i=my_substrate.theta_i
    n_i2=my_substrate.Nsub
    nt=my_substrate.Nsub
    
    costheta_t=[np.sqrt(1-(n_i[i]*np.sin(theta_i)**2/(nt[i]**2))) for i in range(nb)]
    theta_i2=[np.arccos(costheta_t[i]) for i in range(nb)]
    
    Kxx=[n_i2[i]*np.sin(theta_i2[i]) for i in range(nb)]
    if(not layer.anis):
        # if layer is istoropic the eigen value can be express as :
        def q_f(eps,Kxx):
            return np.sqrt(eps-Kxx**2)
        q=np.array(list(map(q_f,eps,Kxx)))
        def Tp_f(omega,q,eps):
            a0=np.cos(omega/c*d*q)
            temp=np.sin(omega/c*d*q)
            d0=temp*q*1j/eps
            c1=temp*(-1j)/q
            b2=temp*q*(-1j)
            a3=temp*1j*eps/q
            Tp=np.matrix([[a0,0,0,d0],
                        [0,a0,c1,0],
                        [0,b2,a0,0],
                        [a3,0,0,a0]])
            return Tp
        Tp=np.array(list(map(Tp_f,omega,q,eps)))
        return Tp
    else: # if layer is anistoropic 
    #Calculus of the delta matrix#
        def delta_f(eps,Kxx):
            a0=-Kxx*(eps[2,0]/eps[2,2])
            b0=-Kxx*(eps[2,1]/eps[2,2])
            d0=1-(Kxx*Kxx/eps[2,2])
            a2=(eps[1,2]*eps[2,0]/eps[2,2])-eps[1,0]
            b2=Kxx*Kxx-eps[1,1]+(eps[1,2]*eps[2,1]/eps[2,2])
            d2=(Kxx*eps[1,2]/eps[2,2])
            a3=(-eps[0,2]*eps[2,0]/eps[2,2])+eps[0,0]
            b3=-(eps[0,2]*eps[2,1]/eps[2,2])+eps[0,1]
            d3=(-Kxx*eps[0,2]/eps[2,2])
            delta=np.matrix([[a0,b0,0,d0],[0,0,-1,0],[a2,b2,0,d2],[a3,b3,0,d3]])
            return delta
        delta=np.array(list(map(delta_f,eps,Kxx)))
        #Calculus of the eigenvalues matrix#
        def q_f(delta):
            return np.linalg.eig(delta)[0]
        q=np.array(list(map(q_f,delta)))
    #Calculus of the eigenvalues matrix#
        def beta_f(q,omega):
            a1=np.exp(-d*1j*q[0]*omega/c)/((q[0]-q[1])*(q[0]-q[2])*(q[0]-q[3]))
            b1=np.exp(-d*1j*q[1]*omega/c)/((q[1]-q[2])*(q[1]-q[3])*(q[1]-q[0]))
            c1=np.exp(-d*1j*q[2]*omega/c)/((q[2]-q[3])*(q[2]-q[0])*(q[2]-q[1]))
            d1=np.exp(-d*1j*q[3]*omega/c)/((q[3]-q[0])*(q[3]-q[1])*(q[3]-q[2]))
            beta_0=-q[1]*q[2]*q[3]*a1-q[2]*q[3]*q[0]*b1-q[3]*q[0]*q[1]*c1-q[0]*q[1]*q[2]*d1
            beta_1=((q[1]*q[2])+(q[1]*q[3])+(q[2]*q[3]))*a1+((q[2]*q[3])+(q[2]*q[0])+(q[3]*q[0]))*b1+((q[3]*q[0])+(q[3]*q[1])+(q[0]*q[1]))*c1+((q[0]*q[1])+(q[0]*q[2])+(q[1]*q[2]))*d1
            beta_2=-(q[1]+q[2]+q[3])*a1-(q[2]+q[3]+q[0])*b1-(q[3]+q[0]+q[1])*c1-(q[0]+q[1]+q[2])*d1
            beta_3=a1+b1+c1+d1
            return [beta_0,beta_1,beta_2,beta_3]
        beta=np.array(list(map(beta_f,q,omega)))
        def Tp_f (beta,delta):
            return beta[0]*np.identity(4)+beta[1]*delta+beta[2]*np.dot(delta,delta)+beta[3]*np.dot(delta,np.dot(delta,delta))
        Tp=np.array(list(map(Tp_f,beta,delta)))
    return Tp

#Function to calculate the global transfert matrix of the substrate/layer/outside medium interface
def tb():
    fmatrix=[]
    nTp=[Tpb(my_layers[j]) for j in range(len(my_layers))]
    nt2=n_i
    n_i2=my_substrate.Nsub
    theta_i=my_substrate.theta_i
    nt=my_substrate.Nsub
    def theta_i2_f (n_i,nt):
        return np.arccos(np.sqrt(1-(n_i*np.sin(theta_i)**2/(nt**2))))
    theta_i2=np.array(list(map(theta_i2_f,n_i,nt)))

    def costheta_t_f (n_i2,nt2,theta_i2):
        return np.sqrt(1-(n_i2*np.sin(theta_i2)**2/(nt2**2)))
    costheta_t=np.array(list(map(costheta_t_f,n_i2,nt2,theta_i2)))

    def Lt_f (costheta_t,nt2):
	    return np.matrix([[0,0,costheta_t,0],[1,0,0,0],[-nt2*costheta_t,0,0,0],[0,0,nt2,0]])
    Lt=np.array(list(map(Lt_f,costheta_t,nt2)))

    def Li_f(n_i2,theta_i2):
	    return 1/2*np.matrix([[0,1,-1/(n_i2*np.cos(theta_i2)),0],[0,1,1/(n_i2*np.cos(theta_i2)),0],[1/np.cos(theta_i2),0,0,1/n_i2],[-1/np.cos(theta_i2),0,0,1/n_i2]])
    Li=np.array(list(map(Li_f,n_i2,theta_i2)))

    fmatrix.append(Li)
    for j in range(len(nTp)):
        fmatrix.append(nTp[j])
    fmatrix.append(Lt)
    T=[np.linalg.multi_dot([(fmatrix[j][i]) for j in range(len(fmatrix))])for i in range(nb)]
    return T

#Function to calculate the Jones matrix of the backreflected light beam
def Jtb_Jrb():
    T=tb()
    def Jt_Jr_f(T):
        a=T[0,0]*T[2,2]-T[0,2]*T[2,0]
        #Calculus of the Jones transmittance coefficients at the incident medium Layer+substrate interface#
        Tss=T[2,2]/a
        Tsp=-T[2,0]/a 
        Tps=-T[0,2]/a
        Tpp=T[0,0]/a
        #Calculus of the Jones reflectance coefficients at the incident medium Layer+substrate interface#
        Rpp=(T[0,0]*T[3,2]-T[0,2]*T[3,0])/a
        Rsp=(T[0,0]*T[1,2]-T[0,2]*T[1,0])/a
        Rss=(T[1,0]*T[2,2]-T[1,2]*T[2,0])/a
        Rps=(T[2,2]*T[3,0]-T[2,0]*T[3,2])/a
        Jr=np.matrix([[Rpp,Rps],[Rsp,Rss]])
        Jt=np.matrix([[Tpp,Tps],[Tsp,Tss]])
        return {'Jr':Jr,'Jt':Jt}
    JtJr=np.array(list(map(Jt_Jr_f,T)))
    Jt=[JtJr[i]['Jt'] for  i in range(nb)]
    Jr=[JtJr[i]['Jr'] for  i in range(nb)]
    return {'Jrb':Jr,'Jtb':Jt}

#Calcul of the total transmittance and reflectance after stack and substrate including zeroth, first and second order beam for the reflectance 
# and first order transmitted beam for the transmittance
def Jb(): 
    d2=my_substrate.d
    nt=my_substrate.Nsub
    theta_i=my_substrate.theta_i
    nJ=Jt_Jr()
    nJt=nJ['Jt']

    #Zeroth order of the reflected Jones matrix
    nJr=nJ['Jr']

    nJb=Jtb_Jrb()
    nJtb=nJb['Jtb']
    nJrb=nJb['Jrb']
    ###Calculus of the fresnel coefficient at the uncoated backside of the substrate Nsub/n_i contribution###
    def theta_1_f (n_i,nt):
        return np.arccos(np.sqrt(1-(n_i*np.sin(theta_i)**2/(nt**2))))
    theta_1=np.array(list(map(theta_1_f,n_i,nt)))

    def theta_2_f (nt,theta_1,n_i):
        return np.arcsin(nt*np.sin(theta_1)/n_i)
    theta_2=np.array(list(map(theta_2_f,nt,theta_1,n_i)))

    def Rbp_f(n_i,theta_1,nt,theta_2):
        return (n_i*np.cos(theta_1)-nt*np.cos(theta_2))/(n_i*np.cos(theta_1)+nt*np.cos(theta_2))
    Rbp=np.array(list(map(Rbp_f,n_i,theta_1,nt,theta_2)))

    def Rbs_f(n_i,theta_1,nt,theta_2):
        return (nt*np.cos(theta_1)-n_i*np.cos(theta_2))/(nt*np.cos(theta_1)+n_i*np.cos(theta_2))
    Rbs=np.array(list(map(Rbs_f,n_i,theta_1,nt,theta_2)))
    
    def Tbp_f(n_i,theta_1,nt,theta_2):
        return (2*nt*np.cos(theta_1))/(n_i*np.cos(theta_1)+nt*np.cos(theta_2))
    Tbp=np.array(list(map(Tbp_f,n_i,theta_1,nt,theta_2)))

    def Tbs_f(n_i,theta_1,nt,theta_2):
        return (2*nt*np.cos(theta_1))/(nt*np.cos(theta_1)+n_i*np.cos(theta_2))
    Tbs=np.array(list(map(Tbs_f,n_i,theta_1,nt,theta_2)))

    Jf=[np.matrix([[Tbp[i],0],[0,Tbs[i]]]) for i in range(nb)]
    rsa=[np.matrix([[Rbp[i],0],[0,Rbs[i]]]) for i in range(nb)]
    
    def Kxx_f(n_i):
        return n_i*np.sin(theta_i)
    Kxx=np.array(list(map(Kxx_f,n_i)))
    def q_f(nt,Kxx):
        return np.sqrt(nt**2-Kxx**2)
    q=np.array(list(map(q_f,nt,Kxx)))
    
    k0=omega/c
    def Jtsub_f(k0,q):
        a=np.exp(1j*k0*d2*q)
        return np.matrix([[a,0],[0,a]])
    Jtsub=np.array(list(map(Jtsub_f,k0,q)))
    
    #Zeroth order transmitted Jones matrix
    def Tt_f(Jf,Jtsub,nJt):
        return np.dot(Jf,np.dot(Jtsub,nJt))
    Tt=np.array(list(map(Tt_f,Jf,Jtsub,nJt)))
    
    #First order reflected Jones matrix
    def Rt_f(nJtb,rsa,Jtsub,nJt):
        return np.dot(nJtb,np.dot(Jtsub,np.dot(rsa,np.dot(Jtsub,nJt))))
    Rt=np.array(list(map(Rt_f,nJtb,rsa,Jtsub,nJt)))

    #Second order reflected Jones matrix
    def Rt2_f(nJtb,rsa,Jtsub,nJt,nJrb):
        return np.dot(nJtb,np.dot(Jtsub,np.dot(rsa,np.dot(Jtsub,np.dot(nJrb,np.dot(Jtsub,np.dot(rsa,np.dot(Jtsub,nJt))))))))
    Rt2=np.array(list(map(Rt2_f,nJtb,rsa,Jtsub,nJt,nJrb)))
    return {'JT':Tt,'JR':nJr,'JR2':Rt,'JR3':Rt2}

#Function to calculate reflectance and transmittance from Jones matrix taking into account zeroth order for transmitted beam and zeroth, first and second order for the reflected beam 
def T_R():
    J=Jb()
    Tt=J['JT']
    Rt=J['JR']
    Rt2=J['JR2']
    Rt3=J['JR3']
    def Tp_f(Tt):
        return np.real(Tt[0,0]*np.conjugate(Tt[0,0]))
    Tp=np.array(list(map(Tp_f,Tt)))
    def Ts_f(Tt):
        return np.real(Tt[1,1]*np.conjugate(Tt[1,1]))
    Ts=np.array(list(map(Ts_f,Tt)))

    def Rp_f(Rt,Rt2,Rt3):
        return np.real((Rt[0,0]*np.conjugate(Rt[0,0]))+(Rt2[0,0]*np.conjugate(Rt2[0,0]))+(Rt3[0,0]*np.conjugate(Rt3[0,0])))
    Rp=np.array(list(map(Rp_f,Rt,Rt2,Rt3)))

    def Rs_f(Rt,Rt2,Rt3):
        return np.real((Rt[1,1]*np.conjugate(Rt[1,1]))+(Rt2[1,1]*np.conjugate(Rt2[1,1]))+(Rt3[1,1]*np.conjugate(Rt3[1,1])))
    Rs=np.array(list(map(Rs_f,Rt,Rt2,Rt3)))
    return {'Ts':Ts,'Tp':Tp,'Tunpol':1/2*(np.asarray(Tp)+np.asarray(Ts)),'Rs':Rs,'Rp':Rp,'Runpol':1/2*(np.asarray(Rp)+np.asarray(Rs))}

#Function to calcultate ellipsometric angle psi and delta in radian taking into account the substrate backside contribution
def psidelb():
    J=Jb()
    J1=J['JR']
    J2=J['JR2']
    J3=J['JR3']
    prp=[np.absolute(J1[i][0,0]) for i in range(nb)]
    prs=[np.absolute(J1[i][1,1]) for i in range(nb)]
    prp2=[np.absolute(J2[i][0,0]) for i in range(nb)]
    prs2=[np.absolute(J2[i][1,1]) for i in range(nb)]
    prp3=[np.absolute(J2[i][0,0]) for i in range(nb)]
    prs3=[np.absolute(J2[i][1,1]) for i in range(nb)]
    
    drp= [J1[i][0,0] for i in range(nb)]
    drs= [J1[i][1,1] for i in range(nb)]
    drp2= [J2[i][0,0] for i in range(nb)]
    drs2= [J2[i][1,1] for i in range(nb)]
    drp3= [J3[i][0,0] for i in range(nb)]
    drs3= [J3[i][1,1] for i in range(nb)]
    def delta_f(drp,drp2,drp3,drs,drs2,drs3):
        return -(np.real(drp*np.conjugate(drs))+np.real(drp2*np.conjugate(drs2))+np.real(drp3*np.conjugate(drs3)))/np.sqrt((np.absolute(drp)**2+np.absolute(drp2)**2+np.absolute(drp3)**2)*(np.absolute(drs)**2+np.absolute(drs2)**2+np.absolute(drs3)**2))
    delta=np.array(list(map(delta_f,drp,drp2,drp3,drs,drs2,drs3)))
    
    def psi_f(prp,prp2,prp3,prs,prs2,prs3):
        return np.arctan2(np.sqrt(prp**2+prp2**2+prp3**2), np.sqrt(prs**2+prs2**2+prs3**2))
    psi=np.array(list(map(psi_f,prp,prp2,prp3,prs,prs2,prs3)))
    return {'psi':np.tan(psi),'delta':delta}