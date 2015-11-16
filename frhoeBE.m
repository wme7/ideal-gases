function psi = frhoeBE(z,n,rho,e,h)
psi = (( Polylog(n/2, z)).^((n+2)/n))./Polylog((n+2)/2, z)-(h^2)*n*rho^(2/n)/(4*pi*e);