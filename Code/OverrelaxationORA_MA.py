from __future__ import annotations

import numpy as np
from puwr import tauint
from dataclasses import dataclass
import time

def WrapAngles(Theta: np.ndarray) -> np.ndarray:
    # Project angles back to interval [-pi, pi)
    return (Theta + np.pi) % (2.0 * np.pi) - np.pi # % is the modulus

@dataclass 
class Lattice2D: #Defining/declaring Lattice parameters Nt-by-Nx
    Nt: int
    Nx: int

    @property
    def Shape(self) -> tuple[int, int]:
        return self.Nt, self.Nx

    @property
    def Volume(self) -> int:  
        return self.Nt * self.Nx

@dataclass
class U1Lattice2D:
    Lattice: Lattice2D #Calling the Lattice by it's parameters/Geometry as defined above (NtxNx)
    Theta:   np.ndarray #Defining/introducing Theta as an array

    @classmethod
    def ColdStart(cls, Lattice: Lattice2D) -> "U1Lattice2D": #Defining the ColdStart
        Theta = np.zeros((Lattice.Nt, Lattice.Nx, 2), dtype = np.float64)
        return cls(Lattice, Theta)

    @classmethod 
    def HotStart(cls, Lattice: Lattice2D, prng: np.random.Generator) -> "U1Lattice2D": #Defining HotStart
        Theta = prng.uniform(low = -np.pi, high = np.pi, size = (Lattice.Nt, Lattice.Nx, 2))
        return cls(Lattice, Theta)

    def Copy(self) -> "U1Lattice2D":
        return U1Lattice2D(self.Lattice, self.Theta.copy())

def StapleSum(Field: U1Lattice2D, mu: int) -> np.ndarray:
    if mu not in (0, 1):
        raise ValueError("Directional index mu must be either 0 or 1!")
    nu = 1 - mu
    forward_staple  = (Field.Theta[:, :, nu]
                    +  np.roll(Field.Theta[:, :, mu], shift = -1, axis = nu)
                    -  np.roll(Field.Theta[:, :, nu], shift = -1, axis = mu))
    backward_staple = (-np.roll(Field.Theta[:, :, nu], shift = 1, axis = nu)
                    +   np.roll(Field.Theta[:, :, mu], shift = 1, axis = nu)
                    +   np.roll(np.roll(Field.Theta[:, :, nu], shift = 1, axis = nu), shift = -1, axis = mu))
    return np.exp(1j * forward_staple) + np.exp(1j * backward_staple)

def MetropolisSweep(Field: U1Lattice2D, beta: float, stepsize: float, prng: np.random.Generator, n_hits: int = 8) -> float:
    Nt, Nx = Field.Lattice.Shape
    grid_t, grid_x = np.indices((Nt, Nx))
    parity         = (grid_t + grid_x) % 2
#in checkerboard (parity) update, we're only updating the independent sites
#when updating site x,its neighors don't change, so the staple at x also doesn't change
#to implement the Multi hit Metropolis, we want to propose multiple updates to a site in a row without having to recalculate the staple, the motivation is that it will help improve acceptance rate and decorrelates lattice faster
    # # parity =
    # 0 1 0 1
    # 1 0 1 0
    # 0 1 0 1
    # 1 0 1 0
    # # current_parity = 0
    # # parity == current_parity
    # 1 0 1 0
    # 0 1 0 1
    # 1 0 1 0
    # 0 1 0 1
    # # current_parity = 1
    # # parity == current_parity
    # 0 1 0 1
    # 1 0 1 0
    # 0 1 0 1
    # 1 0 1 0

    accepted_proposals = 0
    total_proposals    = 0

    # This only work correctly when both Nx and Nt are even numbers
    if Field.Lattice.Nt % 2 != 0 or Field.Lattice.Nx % 2 != 0:
        raise ValueError("MetropolisSweep requires both Nt and Nx to be even!")

    for mu in (0, 1):
        for current_parity in (0, 1):
            parity_mask    = (parity == current_parity)
        
            theta_old      = Field.Theta[:, :, mu].copy()
            theta_proposal = WrapAngles(theta_old + prng.uniform(low = -stepsize, high = stepsize, size = (Nt, Nx)))
            staple = StapleSum(Field, mu)
            # Compute action difference
            DeltaS = -beta * np.real((np.exp(1j * theta_proposal) - np.exp(1j * theta_old)) * np.conjugate(staple))
            # Updates are only accepted for sites where there parity is correct and where the accept-reject step is fulfilled
            accept_mask = parity_mask & ((DeltaS <= 0.0) | (prng.uniform(low = 0.0, high = 1.0, size = (Nt, Nx)) < np.exp(-DeltaS)))

            # Update field with new values where appropriate
            Field.Theta[:, :, mu] = np.where(accept_mask, theta_proposal, theta_old)
            accepted_proposals += np.count_nonzero(accept_mask)
            total_proposals    += np.count_nonzero(parity_mask)
    # Return acceptance rate
    return accepted_proposals / total_proposals
    
    
    
def OverrelaxationSweep(Field: U1Lattice2D) -> None:
    Nt, Nx         = Field.Lattice.Shape
    grid_t, grid_x = np.indices((Nt, Nx))
    parity         = (grid_t + grid_x) % 2
    
    if Field.Lattice.Nt % 2!= 0 or Field.Lattice.Nx % 2 != 0:
        raise ValueError("OverrelaxationSweep rewuires both Nt and Nx to be even!")
        
    for mu in (0,1):
        for current_parity in (0,1):
            parity_mask = (parity == current_parity)
            
            staple = StapleSum(Field, mu) #Getting the complex staple
            
            #Extracting the staple phase phi
            staple_phase = np.angle(staple)
             
            theta_old = Field.Theta[:, :, mu] #Extract old theta
            
            #Applying the deterministic ORA equation: phi' = 2*phi  - theta
            theta_proposal = WrapAngles(2 * staple_phase -theta_old)
            
            #Updating only the sites in the current checkerboard parity
            Field.Theta[:, :, mu] = np.where(parity_mask, theta_proposal, theta_old)       
    

# Observables
# - PlaquetteAngles   -> Array(float)
# - Action            -> float
# - Plaquette         -> float
# - WilsonLoop        -> float
# - TopologicalCharge -> float

def PlaquetteAngles(Field: U1Lattice2D) -> np.ndarray:
    # Plaq = exp(i * arg)
    # arg = theta_t + np.roll(theta_x, shift = -1, axis = 0) - np.roll(theta_t, shift = -1, axis = 1) - theta_x
    theta_plaq = (Field.Theta[:, :, 0]
               +  np.roll(Field.Theta[:, :, 1], shift = -1, axis = 0)
               -  np.roll(Field.Theta[:, :, 0], shift = -1, axis = 1)
               -  Field.Theta[:, :, 1])
    return WrapAngles(theta_plaq)

def Action(Field: U1Lattice2D, beta: float) -> float:
    return float(beta * np.sum(1.0 - np.cos(PlaquetteAngles(Field))))

def AveragePlaquette(Field: U1Lattice2D) -> float:
    return float(np.mean(np.cos(PlaquetteAngles(Field))))

# # TODO: Finish implementation
def WilsonLoop(Field: U1Lattice2D, Lt: int, Lx: int) -> float:
    #Initializing array to hold the accumulated phase for loops at each site
    loop_phase= np.zeros(Field.Lattice.Shape, dtype=np.float64)
    
    #bottom edge: going +x for Lx steps, starting at time t
    for i in range(Lx):
        loop_phase += np.roll(Field.Theta[:,:,1], shift=-i, axis =1)

    #right: going +t for Lt steps, starting at space x+Lx 
    for j in range(Lt):
        loop_phase += np.roll(Field.Theta[:, :, 0],shift =(-j,-Lx), axis = (0,1))
        
    #top: going -x for Lx steps (backward), starting at time t+Lt
    for i in range(Lx):
        loop_phase -= np.roll(Field.Theta[:, :, 1], shift = (-Lt, -i), axis = (0, 1))
        
    #left edge: going -t for Lt steps, starting at space x
    for j in range(Lt):
        loop_phase -= np.roll(Field.Theta[:, :, 0], shift = -j, axis = 0)
    #Wilson loop is the real part of: Re[exp(i * phase)] = cos(phase)
    return float(np.mean(np.cos(loop_phase)))

def TopologicalCharge(Field: U1Lattice2D) -> float:
    return float(1.0 / (2.0 * np.pi) * np.imag(np.sum(np.log(np.exp(1.0j * PlaquetteAngles(Field))))))

#
def BlockBootstrapError(data: np.ndarray, tau_int: float, n_boot: int = 200, prng: np.random.Generator = None) -> float:
    if prng is None:
        prng = np.random.default_rng()
        
    N = len(data)
    #The block size must be an integer (at least 1)
    block_size = max(1, int(np.ceil(2.0 * tau_int)))
    n_blocks = N // block_size
    
    if n_blocks == 0:
        raise ValueError("Data length is too short for the calculated block size.")

    #Truncating the data slightly so it divides into blocks
    trunct_data = data[:n_blocks * block_size].reshape((n_blocks, block_size))
    
    #Generating random block indices for all bootstrap samples (at once)
    rndm_indices = prng.integers(0, n_blocks, size=(n_boot, n_blocks)) #matrix of shape (n_boot, n_blocks)
    
    #Using advanced numpy indexing to build the bootstrap samples
    #and immediately computing the mean for both the blocks and measurements
    boot_means = np.mean(trunct_data[rndm_indices], axis=(1, 2))
    
    #Std of the bootstrap
    return float(np.std(boot_means))

if __name__ == "__main__":  #What should we call this later?
    rng = np.random.default_rng()

    Nt   = 20
    Nx   = 20
    beta = 5

    stepsize         = 1
    n_updates        = 20000
    n_thermal        = 1000 #number of sweeps to reach equilibrium
    
    print(f"{Nt}x{Nx} lattice at beta ={beta}")
    
    #Initializing the lattice geometry; I started with a Cold Start
    lattice= Lattice2D(Nt,Nx)
    field = U1Lattice2D.ColdStart(lattice) # cold start
    #field = U1Lattice2D.HotStart(lattice, rng) #this is for the hot start
    
    # Perform thermalization updates
    print(f"Starting thermalization ({n_thermal} sweeps)")
    for _ in range(n_thermal):
    #Calling the sweep to update the field, but I haven't saved the observables yet
        MetropolisSweep(field,beta,stepsize,rng)
    print("Thermalization complete")
    

    # Allocate arrays for each observable measurement
    P_measurements = np.zeros(n_updates, dtype = np.float64)
    Q_measurements = np.zeros(n_updates, dtype = np.float64)
    acc_rates      = np.zeros(n_updates, dtype = np.float64)
    Wnxn_measurements = np.zeros(n_updates, dtype = np.float64)
    
    n_ORA = 5 #I want to increase this to test the efficiency of ORA
    # Update loop
        # Perform updates and measurements
    print(f"\nStarting main measurement loop ({n_updates} sweeps)")
    t0 =time.time()
    for i in range(n_updates): ##running the main simulations, it looks much more efficient than our MATLAB script
        #updating the lattice and recording the acceptance rate
        acc_rates[i]= MetropolisSweep(field, beta, stepsize, rng)
        
        for _ in range(n_ORA):
            OverrelaxationSweep(field)
        
        #measure and store observables
        P_measurements[i] = AveragePlaquette(field)
        Q_measurements[i] = TopologicalCharge(field)
        
        Wnxn_measurements[i] = WilsonLoop(field, 1, 1)
        #printing progress update every 100 iterations to ensure code is stil running
        #if (i+1) %100 ==0:
            #print(f"Completed {i+1}/{n_updates} sweeps")
    t1 = time.time()
    total_time= t1 - t0
    print(f"\nMeasurement loop complete ({total_time} seconds)")
    print("Measurement loop complete")
    
    
    #Print mean and naive error estimate
    meanP_naive = np.mean(P_measurements)
    errP_naive = np.std(P_measurements)/np.sqrt(n_updates)
    meanQ_naive = np.mean(Q_measurements)
    errQ_naive  = np.std(Q_measurements)/np.sqrt(n_updates)
    
    #Computing topological charge susceptibility
    V = Nt*Nx #Volume
    Q_susc = np.mean(Q_measurements**2) /V #Q susceptibility
    
    mean_acc = np.mean(acc_rates) #average acceptance rate over all sweeps
    mean_Wnxn = np.mean(Wnxn_measurements)
    err_Wnxn  = np.std(Wnxn_measurements) / np.sqrt(n_updates)
    
    #Calculating integrated autocorrelation time for Plaquette
    #tauint returns a tuple containing autocorrelation time ,error,  optimal window, and
    #we should use [[ ]] for the shape, and lambda x: x for required 'f' argument
    meanP_uwerr, errP_uwerr, tauP_uwerr, dtauP_uwerr = tauint([[P_measurements]], lambda x: x)
    print(f"\nAutocorrelation time tau_int for P and Q")
    print(f"Plaquette tau_int: {tauP_uwerr:.2f}") 
    
    #calculating tau_int for topological charge
    meanQ_uwerr, errQ_uwerr, tauQ_uwerr, dtauQ_uwerr = tauint([[Q_measurements]], lambda x: x)
    print(f"Topological Charge tau_int: {tauQ_uwerr:.2f}")
    
    # Printing mean and bootstrap error estimates
    meanP_boot = np.mean(P_measurements)
    errP_boot  = BlockBootstrapError(P_measurements, tauP_uwerr, n_boot=500, prng=rng)
    
    meanQ_boot = np.mean(Q_measurements)
    errQ_boot  = BlockBootstrapError(Q_measurements, tauQ_uwerr, n_boot=500, prng=rng)
    

    
    print(f"n_ORA sweeps: {n_ORA}")
    print(f"\nAcceptance rate= {mean_acc:.2f}")
    print(f"Step size= {stepsize}")
    print(f"Q susceptibility: {Q_susc}")
    print(f"Wilson Loop (1x1): {mean_Wnxn:.6f} ± {err_Wnxn:.6f}")
    
    print(f"\nFinal results for average Plaquette and Q")
    
    print(f"AvgPlaquette (Naive): {meanP_naive:.6f} ± {errP_naive}")
    print(f"AvgPlaquette (Bootstrap): {meanP_boot:.6f} ± {errP_boot:.6f}")
    print(f"AvgPlaquette (Gamma): {meanP_uwerr:.6f} ± {errP_uwerr:.6f}")
    
    
    print(f"\nQ (Naive): {meanQ_naive:.6f} ± {errQ_naive}")
    print(f"Q (Boot): {meanQ_boot:.6f} ± {errQ_boot:.6f}")
    print(f"Q (Gamma): {meanQ_uwerr:.6f} ± {errQ_uwerr:.6f}")
    