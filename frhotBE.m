function psi = frhotBE(z,n,rho,t,h)
psi = rho*h^n./(2*pi*t).^(n/2)-BE(n/2,z);