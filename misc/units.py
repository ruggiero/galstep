BOLTZMANN = 1.3806e-16
PROTONMASS = 1.6726e-24
HYDROGEN_MASSFRAC = 0.76  
GAMMA = (5.0 / 3)   
GAMMA_MINUS1 = (GAMMA - 1)

G = 43007.1

cm_in_km = 1e-5 
erg_in_J = 1e-7
erg_in_kev = 6.2415e8


# argument: temperature, in Kelvin
# returns : internal energy per unit mass, in (km/s)^2
def temp_to_internal_energy(temp):
    # mean molecular weight depends on temperature
    meanweight = mean_weight( temp )

    # internal energy per unit mass (cgs)
    internal_energy = ((1.0/meanweight) * (1.0/GAMMA_MINUS1) *
                       (BOLTZMANN/PROTONMASS) * temp)
  
    # in (km/s)^2
    internal_energy = internal_energy * cm_in_km * cm_in_km
  
    return internal_energy

# argument: temperature, in Kelvin
# returns : energy, in keV
def temp_to_kev(temp):
    return BOLTZMANN * erg_in_kev * temp;

# argument: internal energy per unit mass, in (km/s)^2
# returns : temperature, in Kelvin
def internal_energy_to_temp(internal_energy):
    # energy in cgs:
    internal_energy = internal_energy / cm_in_km / cm_in_km
    
    meanweight_n = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC) # neutral gas
    meanweight_i = 4.0 / (3 + 5 * HYDROGEN_MASSFRAC) # fully ionized gas
    
    temp_n = (internal_energy * meanweight_n * GAMMA_MINUS1 *
              PROTONMASS/BOLTZMANN)
    temp_i = (internal_energy * meanweight_i * GAMMA_MINUS1 * 
              PROTONMASS/BOLTZMANN)
    
    #temp_n will result larger than temp_i
    if (temp_i > 1e4):
        temp = temp_i
    else:
        if (temp_n < 1e4):
            temp = temp_n
        else:
            temp = 0.5 * (temp_i+temp_n)
    return temp;    

# argument: temperature, in Kelvin
# returns : mean molecular weight
def mean_weight(temp):
    if ( temp < 1e4 ):
        meanweight = 4.0 / (1 + 3*HYDROGEN_MASSFRAC) # neutral gas
    else:
        meanweight = 4.0 / (3 + 5*HYDROGEN_MASSFRAC) #fully ionized gas
    return meanweight;
