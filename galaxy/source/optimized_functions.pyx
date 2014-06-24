from libc.math cimport acos, cos, cosh, exp, sin
from numpy import pi, inf
from scipy.integrate import quad


cdef double G = 44923.53
cdef double M, Rd, z0


def phi_disk(double rho, double z, double Mf, double Rdf, double z0f):
    global M, Rd, z0
    cdef double sum_, r, theta
    cdef double t1, t2
    M = Mf
    Rd = Rdf
    z0 = z0f
    r = (rho**2 + z**2)**0.5
    if(r == 0):
        r = 0.1
        theta = 0
    else:
        theta = acos(z / r)
    sum_ = 0.0
    for l in [0, 2, 4, 6]:
        t1 = (1/r**(l+1)) * quad(phi_integrand_1, 0, r, args=(l), limit=200, full_output=-1)[0]
        t2 = r**l * quad(phi_integrand_2, r, inf, args=(l), full_output=-1)[0]
        sum_ += Pl(cos(theta), l) * (t1 + t2)
    return -2*pi*G * sum_ 


cdef double Pl(double x, int l):
    if(l == 0):
        return 1   
    elif(l == 1):
        return x
    elif(l == 2):   
        return 0.5*(3*x**2-1)
    elif(l == 3):
        return 0.5*(5*x**3-3*x)
    elif(l == 4):
        return 0.125*(35*x**4-30*x**2+3)
    elif(l == 5):
        return 0.125*(63*x**5-70*x**3+15*x)
    elif(l == 6):
        return 0.0625*(231*x**6-315*x**4+105*x**2-5)
    elif(l == 7):
        0.0625 * (429*x**7-693*x**5+315*x**3-35*x)
    elif(l == 8):
        0.0078125 * (6435*x**8-12012*x**6+6930*x**4-1260*x**2+35)
    elif(l == 9):
        0.0078125 * (12155*x**9-25740*x**7+18018*x**5-4620*x**3+315*x)
    elif(l == 10):
        0.00390625*(46189*x**10-109395*x**8+90090*x**6-30030*x**4+3465*x**2-63)


def phi_integrand_1(double r, int l):
    return rho_l(r, l) * r**(l+2)


def phi_integrand_2(double r, int l):
    return rho_l(r, l) / (r**(l-1))


cdef double rho_l(double r, int l):
    return quad(rho_l_integrand, 0, pi, args=(r, l))[0]


def rho_l_integrand(double theta, double r, int l):
    cdef double rho, z
    rho = r * sin(theta)
    z = r * cos(theta)
    return M / (4*pi*z0*Rd**2) * ((1 / cosh(z/z0)**2) * 
        exp(-rho/Rd) * sin(theta) * Pl(cos(theta), l))
