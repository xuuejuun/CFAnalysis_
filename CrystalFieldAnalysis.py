# -*- coding: utf-8 -*-
# Author: GONG Xuejun 龚雪君
# Date: 2019.10.29

from sympy.physics.quantum.cg import CG
import scipy.constants as sc
import numpy as np
from sympy import S
from math import sqrt
import cmath
import pickle



class CFAnalysis():
    def __init__(self,title,lthf,Akqlist,L,SS,J,Hzee,axes,lamda,ks):
        self.title = title
        self.ge = sc.physical_constants["electron g factor"][0]*(-1)
        self.lthf = lthf
        self.Hzee = Hzee
        self.ks = ks
        self.Akqlist = Akqlist
        self.axes = axes
        self.lamda = lamda
        self.L = L
        self.mL = list(range(-self.L,self.L+1,1))
        self.SS = SS
        self.mS = list(np.arange(-self.SS,self.SS+1,1))
        self.J = J
        self.mJ = list(np.arange(-self.J,self.J+1,1))
        #########################################
        self.bra, self.ket = self.get_braket_for_L() 
        self.Sbra, self.Sket = self.get_braket_for_S()
        self.Lmatrix = self.get_L_matrix()
        self.Smatrix = self.get_S_matrix()
        self.Akq = self.AkqCalculator()
        self.Okq = self.OkqCalculator()
        self.JM = self.Decompose_JM()
        self.Lx, self.Ly, self.Lz, self.Sx, self.Sy, self.Sz = self.transform_csv_into_xyz()
        self.Hcf = self.sum_up_Hcf()
        self.H ,self.Hso = self.construct_Hamiltonian()
        self.new_H, self.mu, self.Hzeeman = self.introduce_magnetic_field()
        # self.Jm = self.sum_up_JM()
        
        self.eigenvalues, self.eigenvectors = self.calculate_eigenCF()
        self.ZeeSplitting = self.calculate_zeeman_splitting()
        self.Jprojections = self.calculate_J_projection()
        self.g_tensor = self.calculate_g_tensor()
    

    def Decompose_JM(self):
        """
        Decompose J into the basis of |L ml,S, mS>.
        """
        J = self.J
        mJ = self.mJ
        L = self.L
        SS = self.SS
        mL = self.mL
        mS = self.mS
        ket = self.ket
        bra = self.bra
        Sket = self.Sket
        Sbra = self.Sbra
        # print(J,mJ,L,mL,SS,mS,ket,bra,Sket,Sbra)
        JM = {}
        JM[J]={}
        for mj in mJ:
            term = []
            for ml in mL:           
                for ms in mS:
                    subterm = 0.0
                    cgcoef = CG(L,ml,SS,ms,J,mj).doit()
                    # basis = np.kron(np.dot(ket[ml+3],bra[ml+3]),np.dot(Sket[int(ms+1/2)],Sbra[int(ms+1/2)]))
                    basis = np.kron(np.dot(ket[ml+int(L)],bra[ml+int(L)]),np.dot(Sket[int(ms+SS)],Sbra[int(ms+SS)]))
                    subterm += cgcoef*basis
                    term.append(subterm)
            sum_term = sum(term)
            JM[J][mj]=sum_term
        return JM

    def get_braket_for_L(self):
        """
        eg:L=3
        =======================
        for kets:
        -----------------------
        |3,3> =  0
                 0
                 0
                (0)
                 0
                 0
                 1
        -----------------------
        |3,2> =  0
                 0
                 0
                (0)
                 0
                 1
                 0 
        .
        .
        .
        
        |3,-3> = 1
                 0
                 0
                (0)
                 0
                 0
                 0 
        =======================
        for bras:
        -----------------------
        <3,3| = (0 0 0 0 0 0 1)
        -----------------------
        <3,2| = (0 0 0 0 0 1 0)
        -----------------------
        .
        .
        .
        
        <3,-3| = (1 0 0 0 0 0 0)
        -----------------------
        """
        L = self.L

        basis = np.eye(2*L+1,2*L+1)
        # print(basis, basis.shape)
        bra = []
        for label in range(2*L+1):
            bra.append(np.array([basis[label]]))
    
        ket = []
        for label in range(2*L+1):
            ket.append(bra[label].T)
        return bra, ket

    def get_braket_for_S(self):
        """
        eg:S=1/2
        =======================
        for kets:
        -----------------------
        |1/2,-1/2> =  1
                     (0)
        -----------------------
        |1/2,1/2> =   0
                     (1)
        =======================
        for bras:
        -----------------------
        <1/2,-1/2| = (1 0)
        -----------------------
        <1/2,1/2| = (0 1)
        -----------------------
        """
        SS = self.SS
        basis = np.eye(int(2*SS+1),int(2*SS+1))
        Sbra = []
        for label in range(int(2*SS+1)):
            Sbra.append(np.array([basis[label]]))
        # print(Sbra)
    
        Sket = []
        for label in range(int(2*SS+1)):
            Sket.append(Sbra[label].T)
        return Sbra, Sket

    def AkqCalculator(self):
        """
        convert Akqlist into a dictionary of Akq.
        the structure of Akq:
        Akq{"2": *,"4": *, "6": *}
        * is a subdictionary. eg: when key = 2:
        * = {"-2":*, "-1":*, "0":*, "1":*, "2":*}
        """
        ks = self.ks
        Akqlist = self.Akqlist
        # print(ks)
        # print(type(ks[0]))
        Akq = {}
        n = 0
        for k in ks:
            Akq[str(k)] = {} 
            for q in range(-k,k+1,1):
                Akq[str(k)][str(q)] = Akqlist[n]
                n+=1
        return Akq
    
    def OkqCalculator(self):
        """
        calculate Okq by Clebsh-Gordan Coefficients.
        return Okq in the form of a dictionary.
        similar structure with Akq.
        """ 
        # generating ml pairs.
        L = self.L
        ks = self.ks
        ket = self.ket
        bra = self.bra

        Lpool = list(range(-L,L+1,1))
        Lpair = []
        for l in Lpool:
            for ll in Lpool:
                Lpair.append([l, ll])        
        # build Okq operator for every k and q.
        Okq = {}
        for k in ks:
            Okq[str(k)] = {}
            for q in range(-k,k+1,1):
                term = 0.0
                for ml in Lpair:
                    ml1, ml2 = ml[0], ml[1]
                    upper_coef = CG(L,ml1,k,q,L,ml2).doit()
                    lower_coef = CG(L,L,k,0,L,L).doit()
                    coef = upper_coef/lower_coef
                    product = np.dot(ket[ml2+int(L)], bra[ml1+int(L)])
                    term += coef*product
                Okq[str(k)][str(q)] = term
        return Okq

    def HcfCalculator(self):
        """
        Calculate crystal field hamitonian.
        """
        Akq = self.Akq 
        Okq = self.Okq

        Hcf = {}
        for k in Akq.keys():
            Hcf[k] = {}
            for q in Akq[k].keys():
                Hcf[k][q] = Akq[k][q]*Okq[k][q]
        return Hcf   

    def get_L_matrix(self):
        """
        get L matrix written in covariant spherical vectors.
        return: Lmate in a dictionary.
        Lmate{"-1": *, "0": *, "1": *}
        """
        
        # generating ml pairs.
        Lpool = list(range(-self.L,self.L+1,1))
        Lpair = []
        for l in Lpool:
            for ll in Lpool:
                Lpair.append([l, ll])  
        # initialize the structure of L matrix elements
        L = self.L
        ket = self.ket
        bra = self.bra

        Lmate={}
        for mu in range(-1,2,1):
            Lmate[str(mu)]={}
            term = 0.0
            for ml in Lpair:
                ml1, ml2 = ml[0], ml[1]
                upper_coef = CG(L,ml1,1,mu,L,ml2).doit()
                lower_coef = CG(L,L,1,0,L,L).doit()
                coef = L*upper_coef/lower_coef
                # shift pair from[-L,L] to [0,2L]
                product = np.dot(ket[ml2+int(L)], bra[ml1+int(L)])
                term += coef*product
                Lmate[str(mu)]= term
        return Lmate

    def get_S_matrix(self):
        """
        get S matrix written in covariant spherical vectors.
        return: Smate in a dictionary.
        Smate{"-1": *, "0": *, "1": *}
        """
    
        # generating ms pairs.
        SS = self.SS
        Sket = self.Sket
        Sbra = self.Sbra

        Spool = list(np.arange(-SS,SS+1,1))
        Spair = []
        for s in Spool:
            for ss in Spool:
                Spair.append([s, ss])
        # print(Spair)
        # initialize the structure of L matrix elements
        Smate={}
        for mu in range(-1,2,1):
            # print(mu)
            Smate[str(mu)]={}
            term = 0.0
            for ms in Spair:
                ms1, ms2 = ms[0], ms[1]
                # print(ms1,ms2)
                upper_coef = CG(SS,ms1,1,mu,SS,ms2).doit()
                lower_coef = CG(SS,SS,1,0,SS,SS).doit()
                coef = SS*upper_coef/lower_coef
                # print("coef: ", coef)
                product = np.dot(Sket[int(ms2+SS)], Sbra[int(ms1+SS)])
                # print("basis: ", product)
                term += coef*product
                Smate[str(mu)]= term
        return Smate
    
    def transform_csv_into_xyz(self):
        """
        transform from covariant spherical vectors L-1,L0,L+1 
        to cartisian Lx, Ly, Lz.
        also for S
        Reference:
        Page 14,
        Varshalovich, D.A., Moskalev, A.N. & Khersonskii, V.K. 1988, 
        Quantum theory of angular momentum: irreducible tensors, 
        spherical harmonics, vector coupling coefficients, 
        3nj symbols, World Scientific, Singapore.
        """
        Lmatrix = self.Lmatrix
        Smatrix = self.Smatrix
        lamda = self.lamda
        SS = self.SS
        L = self.L

        Lx = np.kron(1/sqrt(2)*(Lmatrix["-1"]-Lmatrix["1"]), np.eye(int(2*SS+1),int(2*SS+1)))
        Ly = np.kron(1j/(sqrt(2))*(Lmatrix["-1"]+Lmatrix["1"]),np.eye(int(2*SS+1),int(2*SS+1)))
        Lz = np.kron(Lmatrix["0"], np.eye(int(2*SS+1),int(2*SS+1)))
        #################################
        Sx = np.kron(np.eye(int(2*L+1),int(2*L+1)), 1/sqrt(2)*(Smatrix["-1"]-Smatrix["1"]))
        Sy = np.kron(np.eye(int(2*L+1),int(2*L+1)), 1j/(sqrt(2))*(Smatrix["-1"]+Smatrix["1"]))
        Sz = np.kron(np.eye(int(2*L+1),int(2*L+1)), Smatrix["0"])
        #################################
        Hso = lamda*(Lx.dot(Sx) + Ly.dot(Sy) + Lz.dot(Sz))
        Hso = np.array(Hso, dtype=complex)
        #################################
        # eigenvalue,eigenvector = np.linalg.eig(Hso)
        # print("Hso: ")
        # print(eigenvalue)
        return Lx, Ly, Lz, Sx, Sy, Sz

        
    def introduce_magnetic_field(self):
        """
        Introducing a weak magnetic field to the original Hamitonian.So that
        there will be a slight splitting between the degenerated states.
        
        """
        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz
        Sx = self.Sx
        Sy = self.Sy
        Sz = self.Sz
        ge = self.ge
        Hzee = self.Hzee
        axes = self.axes
        H = self.H
        
        

        # magnetic momentum
        mu = {}
        mu['x'] = -(Lx + ge*Sx)
        mu['y'] = -(Ly + ge*Sy) 
        mu['z'] = -(Lz + ge*Sz)
        # print("mu_x", mu['x'])
        # magnetic field
        Hzeeman = {}
        Hzeeman['x'] = mu['x']*Hzee*axes[0]
        Hzeeman['y'] = mu['y']*Hzee*axes[1] 
        Hzeeman['z'] = mu['z']*Hzee*axes[2]
        new_H = np.array(H + Hzeeman['x'] + Hzeeman['y'] + Hzeeman['z'],dtype=complex)
        eigenvals, eigenvecs = np.linalg.eigh(new_H)

        return new_H,mu,Hzeeman
    
    
    def calculate_zeeman_splitting(self):
        J = self.J
        mu = self.mu
        axes = self.axes
        U = self.eigenvectors

        Hzeeman = self.Hzeeman['x'].copy() + self.Hzeeman['y'].copy() + self.Hzeeman['z'].copy()

        result = np.array(U.conj().T.dot(Hzeeman*10000000).dot(U), dtype=complex)
        cf_part = result[0:int(2*J+1), 0:int(2*J+1)].copy()  
        eigenvals,eigenvecs = np.linalg.eigh(cf_part)
        print("-------------------Zeeman Splitting------------------") 
        print(*eigenvals, sep="\n")
        # print("-"*40) 
        return eigenvals
    
    
    def calculate_g_tensor(self):
        """
        calculate g parameters for the ground KD states
        """
        mu = self.mu
        eigenvectors = self.eigenvectors
        # effective S
        S = 1/2 

        factor = 12/((2*S+1)**3-(2*S+1))

        mu_x = eigenvectors.conj().T.dot(mu['x']).dot(eigenvectors)
        mu_y = eigenvectors.conj().T.dot(mu['y']).dot(eigenvectors)
        mu_z = eigenvectors.conj().T.dot(mu['z']).dot(eigenvectors)
        
        Mu = []
        Mu.append(np.array(mu_x[0:2,0:2],dtype=complex))
        Mu.append(np.array(mu_y[0:2,0:2],dtype=complex))
        Mu.append(np.array(mu_z[0:2,0:2],dtype=complex))
        
        print("------------calculating g tensor---------------- ")
        
        g  = []
        A = np.empty((3,3), dtype=complex)
        B = np.empty((3,3), dtype=complex)
        for i in range(3):
            for j in range(3):
                A[i][j]= (np.trace(np.dot(Mu[i],Mu[j])))
                B[i][j]= (np.trace(np.dot(Mu[j],Mu[i])))
    
        C = (1/2)*(A+B)*factor
        w, v = np.linalg.eigh(C)
        print("------eigenvectors of C(main magnetic axes)------")
        print(v)

        for i in range(3):
            gi = cmath.sqrt(w[i])
            g.append(gi)

        print('----------main values of g tensor----------')
        print(*g,sep='\n')
        return g
        

    def sum_up_Hcf(self):
        """
        sum up every components of Hcf. (rank 2, 4, 6).
        """
        ks = self.ks
        Hcf = self.HcfCalculator()
        

        sum_Hcf = []
        for k in ks:
            a = sum(Hcf[str(k)].values())
            sum_Hcf.append(np.array(a,dtype=complex))
        summ_Hcf = sum(sum_Hcf)
        return summ_Hcf

    def sum_up_JM(self):
        """
        sum up all multiplets of Jm
        return Jm
        """
        JM = self.JM
        J = self.J
        Jm = 0.0
        for key in JM[J].keys():
          Jm += JM[J][key]
        return Jm
    

    def calculate_J_projection(self):
        """
        Calculate the projection between Jm and CF states.(in the basis of |LmL,SmS>) 
        """
        JM = self.JM.copy()
        J = self.J
        cfwf = self.eigenvectors.copy()
        Jprojections=[]
        for n in range(len(cfwf)):
            wf_ket = np.array([cfwf[:,n]]).T
            wf_bra = np.array([cfwf[:,n]]).conj()
            projections = []    
            for key in JM[J].keys():
                jket = np.array([JM[J][key].sum(axis=0)]).T 
                jbra = jket.T.conj().copy()
                OP = jket.dot(jbra)
                result = wf_bra.dot(OP.dot(wf_ket))
                projections.append(np.array(result,dtype=complex))
            summ = np.sum(projections)
            Jprojections.append(summ)
        print("-----------J projections on CF states--------------")
        print(*Jprojections, sep="\n")
        return Jprojections


    def calculate_eigenCF(self):
        """
        calculate CF wavefunctions and energy spectrum.
        """
        new_H = self.new_H.copy() 
        eigenvalues, eigenvectors = np.linalg.eigh(new_H)
        print("---------Energy Spectrum of Current H----------")
        print(eigenvalues)
        # self.shift_spectrum(eigenvalues)
        return eigenvalues, eigenvectors



    @staticmethod
    def shift_spectrum(eigenvalue):
        """
        args: eigenvalue: []
        return NONE
        """

        baseline = eigenvalue[0]
        new_eig_vals = [np.real(eig - baseline) for eig in eigenvalue]
        print("-------------shifted energy spectrum-----------")
        print(*new_eig_vals, sep='\n')
        # save energy spectrum in a file.
        # df = open('energy_spectrum_Co_lambda.pickle','ab')
        # pickle.dump(new_eig_vals,df)
        # df.close()
    

    def construct_Hamiltonian(self):
        """
        construct Hamiltonian of crystal field and spin-orbit couping.
        return: H
        """
        Lmatrix = self.Lmatrix 
        Smatrix = self.Smatrix
        Hcf = self.Hcf
        lamda = self.lamda
        SS = self.SS
        lthf = self.lthf

        LS = -np.kron(Lmatrix["-1"],Smatrix["1"]) + np.kron(Lmatrix["0"],Smatrix["0"]) - np.kron(Lmatrix["1"],Smatrix["-1"])
        new_LS = np.array(LS, dtype=float)
        new_Hcf = np.kron(Hcf, np.eye(int(2*SS+1),int(2*SS+1)))
        ##########################################
        if lthf == 'Yes':
            print("*"*40)
            print("Less than half filled system")
            print("*"*40)
            H = new_Hcf + lamda*new_LS
            Hso = lamda*new_LS
        elif lthf == 'No':
            print("*"*40)
            print("More than half filled system")
            print("*"*40)
            H = new_Hcf - lamda*new_LS
            Hso = -lamda*new_LS
        else:
            print("Illegal input. Please check again.")

        eigenvalue, eigenvector = np.linalg.eigh(Hso)
        print('----------energy spectrum of Hso--------------')
        print(eigenvalue)
        eigenvalue, eigenvector = np.linalg.eigh(new_Hcf)
        print('----------energy spectrum of Hcf--------------')
        print(eigenvalue)
        return H, Hso




def main():
    np.set_printoptions(threshold=np.inf) 
    #################################################
    # define the title of the system.
    # -----------------------------------------------
    title = 'Ce' 
    #################################################
    # Is this model less than half filled?
    #------------------------------------------------
    lthf = 'Yes'
    #################################################
    # define angular momentum here.
    #------------------------------------------------
    L = 3
    SS = 1/2
    J = 5/2
    #------------------------------------------------
    # Rank of operator Akq and Okq.
    #------------------------------------------------
    ks = [2,4,6]
    #------------------------------------------------
    # Define spin-orbit coupling parameter lamda here.
    #------------------------------------------------
    lamda = 673.5 # ce compound
    # lamdas = list(range(0,1000,2))
    #------------------------------------------------
    # Introduce a tiny zeeman H here.
    #------------------------------------------------
    Hzee = 0.0000001
    #################################################
    #------------------------------------------------
    # Apply magnetic field along the pesudospin L
    #  direction.
    #------------------------------------------------
    axes = np.array([-0.67416089, -0.73503685, -0.07230443])
    # axes = np.array([0.0, 0.0, 1.0])
    #################################################
    #------------------------------------------------
    # Crystal Field Parameter list.
    #------------------------------------------------
    Akqlist = [+5.90754254386178E+01-1.18615186875176E+02*1j,
      +4.54800460057671E+00-3.35688062687085E+00*1j,
      -1.27580977847726E+02-7.02324225149286E-15*1j,
      -4.54800460057670E+00-3.35688062687085E+00*1j,
      +5.90754254386178E+01+1.18615186875176E+02*1j,
      +1.39571425936749E+02+1.93176098531471E+02*1j,
      -5.84480879742355E-01-2.09122115871930E+00*1j,
      +8.06559481085573E+01-1.62589282078262E+02*1j,
      -6.88668622567206E-01-8.49901744297310E-01*1j,
      -1.74046647829692E+02-4.13898297906542E-15*1j,
      +6.88668622567206E-01-8.49901744297310E-01*1j,
      +8.06559481085573E+01+1.62589282078261E+02*1j,
      +5.84480879742352E-01-2.09122115871930E+00*1j,
      +1.39571425936749E+02-1.93176098531471E+02*1j,
      +4.57181249585863E+00-9.51502956147485E-01*1j,
      +5.67524260733775E-01+1.31543092741297E-01*1j,
      -1.81115676579430E+00-3.34912249075938E+00*1j,
      +1.13854007129954E-01+4.51950008675601E-01*1j,
      -1.60699992889791E+00+3.07932818794209E+00*1j,
      +3.27097313377889E-03-7.25841125095930E-02*1j,
      +3.48041998553132E+00+8.38696089640110E-18*1j,
      -3.27097313377901E-03-7.25841125095930E-02*1j,
      -1.60699992889791E+00-3.07932818794209E+00*1j,
      -1.13854007129954E-01+4.51950008675602E-01*1j,
      -1.81115676579430E+00+3.34912249075938E+00*1j,
      -5.67524260733775E-01+1.31543092741298E-01*1j,
      +4.57181249585863E+00+9.51502956147485E-01*1j]

    #################################################
    #------------------------------------------------
    # Main function. Design your own calculations.
    #------------------------------------------------
    print("J = ", J)
    print("spin orbit coupling constant(lambda) = ", lamda)
    #################################################
 
    Ce = CFAnalysis(title,lthf,Akqlist,L,SS,J,Hzee,axes,lamda,ks) 


if __name__ == "__main__":
    main()
    

"""

import numpy as np 
from CrystalFieldAnalysis import CFAnalysis 
title = 'Ce'  
L = 3 
SS = 1/2 
J = 5/2  
ks = [2,4,6] 
lamda = 673.5   
Hzee = 0.0000001
axes = np.array([-0.67416089, -0.73503685, -0.07230443])
Akqlist = [+5.90754254386178E+01-1.18615186875176E+02*1j, 
            +4.54800460057671E+00-3.35688062687085E+00*1j, 
            -1.27580977847726E+02-7.02324225149286E-15*1j, 
            -4.54800460057670E+00-3.35688062687085E+00*1j, 
            +5.90754254386178E+01+1.18615186875176E+02*1j, 
            +1.39571425936749E+02+1.93176098531471E+02*1j, 
            -5.84480879742355E-01-2.09122115871930E+00*1j, 
            +8.06559481085573E+01-1.62589282078262E+02*1j, 
            -6.88668622567206E-01-8.49901744297310E-01*1j, 
            -1.74046647829692E+02-4.13898297906542E-15*1j, 
            +6.88668622567206E-01-8.49901744297310E-01*1j, 
            +8.06559481085573E+01+1.62589282078261E+02*1j, 
            +5.84480879742352E-01-2.09122115871930E+00*1j, 
            +1.39571425936749E+02-1.93176098531471E+02*1j, 
            +4.57181249585863E+00-9.51502956147485E-01*1j, 
            +5.67524260733775E-01+1.31543092741297E-01*1j, 
            -1.81115676579430E+00-3.34912249075938E+00*1j, 
            +1.13854007129954E-01+4.51950008675601E-01*1j, 
            -1.60699992889791E+00+3.07932818794209E+00*1j, 
            +3.27097313377889E-03-7.25841125095930E-02*1j, 
            +3.48041998553132E+00+8.38696089640110E-18*1j, 
            -3.27097313377901E-03-7.25841125095930E-02*1j, 
            -1.60699992889791E+00-3.07932818794209E+00*1j, 
            -1.13854007129954E-01+4.51950008675602E-01*1j, 
            -1.81115676579430E+00+3.34912249075938E+00*1j, 
            -5.67524260733775E-01+1.31543092741298E-01*1j, 
            +4.57181249585863E+00+9.51502956147485E-01*1j]
Ce = CFAnalysis(title,Akqlist,L,SS,J,Hzee,axes,lamda,ks)       
"""






















