function psi = frhotFD(z,n,rho,t,h)
psi = rho*h^n./(2*pi*t).^(n/2)-FD(n/2,z);