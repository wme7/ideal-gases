function psi = dfrhoeBE(z,n)
Qm =  Polylog(n/2-1, z);
Qo =  Polylog( n/2 , z);
Qp =  Polylog(n/2+1, z);
psi= ((2+n)/n) * (Qo^((2+n)/n-1)*Qm / (z*Qp)) - Qo^(2*(1+n)/n) / (z*Qp^2);