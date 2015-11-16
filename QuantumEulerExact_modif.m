function [x,rho,ux,p,e,z,t,Mach,entro]=QuantumEulerExact_modif(rho1,u1,t1,rho4,u4,t4,tEnd,x,n,h,type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Quantum Gas Exact Riemann Solver
% Coded by Manuel Diaz, IAM, NTU 11.12.2014.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTE:
% A Cavitation Check is incorporated in the code. It further prevents
% plotting for possible but physically unlikely case of expansion shocks. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT VARIABLES: 
% Problem definition: Conditions at time t=0
%   rho1, u1, t1
%   rho5, u4, t4
% 'tEnd' and 'n' are the final solution time and the gas DoFs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define fugacity and Pressure of gas

switch type
    case 'FD' % Find z1, p1, z4 and p4 for FD gas
        z1=0.001; delta=1; it=0;
        while abs(delta) > 1E-6
            z=z1+frhotFD(z1,n,rho1,t1,h)/dfrhotFD(z1,n);
            delta=z-z1; z1=z; %disp(z)
            it=it+1;
        end
        p1 = rho1*t1*FD(n/2+1,z1)/FD(n/2,z1);
        
        z4 = 0.001; delta = 1; it = 0;
        while abs(delta) > 1E-6
            z=z4+frhotFD(z4,n,rho4,t4,h)/dfrhotFD(z4,n);
            delta=z-z4; z4=z; %disp(z)
            it=it+1;
        end
        p4 = rho4*t4*FD(n/2+1,z4)/FD(n/2,z4);
        
    case 'MB'
        p1 = rho1*t1; 
        p4 = rho4*t4;
        
    case 'BE' % Find z1, p1, z4 and p4 for BE gas
        z1=0.001;delta=1;it=0;
        while abs(delta) > 1E-6
            z=z1-frhotBE(z1,n,rho1,t1,h)/dfrhotBE(z1,n); if z > 1; z=0.999; end
            delta = z-z1; z1=z; %disp(z)
            it=it+1;
        end
        p1 = rho1*t1*BE(n/2+1,z1)/BE(n/2,z1);
        
        z4=0.001;delta=1;it=0;
        while abs(delta) > 1E-6
            z=z4-frhotBE(z4,n,rho4,t4,h)/dfrhotBE(z4,n); if z > 1; z=0.999; end
            delta = z-z4; z4=z; %disp(z)
            it=it+1;
        end
        p4 = rho4*t4*BE(n/2+1,z4)/BE(n/2,z4);
end

% Gamma values
gamma=(n+2)/n; alpha=(gamma+1)/(gamma-1);

% Assumed structure of exact solution
%
%    \         /      |con |       |s|
%     \   f   /       |tact|       |h|
% left \  a  /  state |disc| state |o| right
% state \ n /    2    |cont|   3   |c| state
%   1    \ /          |tinu|       |k|   4
%         |           |ity |       | |

PRL = p4/p1;
cright = sqrt(gamma*p4/rho4); 
cleft  = sqrt(gamma*p1/rho1);
CRL = cright/cleft;
MACHLEFT = (u1-u4)/cleft;

% Basic shock tube relation equation (10.51)
f = @(P) (1+MACHLEFT*(gamma-1)/2-(gamma-1)*CRL*(P-1)/sqrt(2*gamma*(gamma-1+(gamma+1)*P)))^(2*gamma/(gamma-1))/P-PRL;

% solve for P = p34 = p3/p4
p34 = fzero(f,3);

p3 = p34*p4;
rho3 = rho4*(1+alpha*p34)/(alpha+p34); 
rho2 = rho1*(p34*p4/p1)^(1/gamma);
u2 = u1-u4+(2/(gamma-1))*cleft*(1-(p34*p4/p1)^((gamma-1)/(2*gamma)));
c2 = sqrt(gamma*p3/rho2);
spos = 0.5+tEnd*cright*sqrt((gamma-1)/(2*gamma)+(gamma+1)/(2*gamma)*p34)+tEnd*u4;

x0 = 0.5;
conpos=x0 + u2*tEnd+tEnd*u4;	% Position of contact discontinuity
pos1 = x0 + (u1-cleft)*tEnd;	% Start of expansion fan
pos2 = x0 + (u2+u4-c2)*tEnd;	% End of expansion fan

% Plot structures
%x = 0:0.002:1; % <--------- now x is defined as an input!
p = zeros(size(x)); 
z = zeros(size(x)); 
ux= zeros(size(x)); 
rho = zeros(size(x));
Mach = zeros(size(x));  
cexact = zeros(size(x));

for i = 1:length(x)
    if x(i) <= pos1
        p(i) = p1;
        rho(i) = rho1;
        ux(i) = u1;
        cexact(i) = sqrt(gamma*p(i)/rho(i));
        Mach(i) = ux(i)/cexact(i);
    elseif x(i) <= pos2
        p(i) = p1*(1+(pos1-x(i))/(cleft*alpha*tEnd))^(2*gamma/(gamma-1));
        rho(i) = rho1*(1+(pos1-x(i))/(cleft*alpha*tEnd))^(2/(gamma-1));
        ux(i) = u1 + (2/(gamma+1))*(x(i)-pos1)/tEnd;
        cexact(i) = sqrt(gamma*p(i)/rho(i));
        Mach(i) = ux(i)/cexact(i);
    elseif x(i) <= conpos
        p(i) = p3;
        rho(i) = rho2;
        ux(i) = u2+u4;
        cexact(i) = sqrt(gamma*p(i)/rho(i));
        Mach(i) = ux(i)/cexact(i);
    elseif x(i) <= spos
        p(i) = p3;
        rho(i) = rho3;
        ux(i) = u2+u4;
        cexact(i) = sqrt(gamma*p(i)/rho(i));
        Mach(i) = ux(i)/cexact(i);
    else
        p(i) = p4;
        rho(i) = rho4;
        ux(i) = u4;
        cexact(i) = sqrt(gamma*p(i)/rho(i));
        Mach(i) = ux(i)/cexact(i);
    end
end
entro = log(p./rho.^gamma);	% entropy
e = p./((gamma-1).*rho);	% internal energy
t = 2/n.*e;                 % temperature

switch type
    case 'FD' % Find z for FD model
        parfor i = 1:length(x)
            zFermi0 = 0.001; delta = 1;
            while abs(delta) > 1E-6
                zFermi=zFermi0+frhoeFD(zFermi0,n,rho(i),e(i),h)/dfrhoeFD(zFermi0,n);
                delta=zFermi-zFermi0; zFermi0=zFermi;
            end
            z(i) = zFermi;
        end
        t = ((rho./(-PolyLog(n/2,-z))).^(2/n))*(h^2)/(2*pi);
        
    case 'MB'
        z = rho*h^n./(2*pi*t).^(n/2);
        
    case 'BE' % Find z for BE model
        parfor i = 1:length(x)
            zBose0 = 0.001; delta = 1;
            while abs(delta) > 1E-6
                zBose=zBose0-frhoeBE(zBose0,n,rho(i),e(i),h)/dfrhoeBE(zBose0,n); if zBose>1; zBose=0.999; end
                delta=zBose-zBose0; zBose0=zBose;
            end
            z(i) = zBose;   
        end
        t = ((rho./( PolyLog(n/2, z))).^(2/n))*(h^2)/(2*pi);
        
end

%% Define Internal Function
    % main quantum functions
    function g = BE(n,z); g =  Polylog(n, z); end
    function g = FD(n,z); g = -Polylog(n,-z); end
    % quantum relations between density-temperature & density-internal energy
    function psi = frhotFD(z,n,rho,t,h); psi = rho*h^n./(2*pi*t).^(n/2)-FD(n/2,z); end
    function psi = frhotBE(z,n,rho,t,h); psi = rho*h^n./(2*pi*t).^(n/2)-BE(n/2,z); end
    %function psi = frhoeFD(z,n,rho,e,h); psi = ((-Polylog(n/2,-z)).^((n+2)/n))./Polylog((n+2)/2,-z)+(h^2)*n*rho^(2/n)/(4*pi*e); end
    %function psi = frhoeBE(z,n,rho,e,h); psi = (( Polylog(n/2, z)).^((n+2)/n))./Polylog((n+2)/2, z)-(h^2)*n*rho^(2/n)/(4*pi*e); end
    % derivatives of the avobe quantum relations
    function psi = dfrhotFD(z,n); psi =  FD(n/2-1,z)/z; end
    function psi = dfrhotBE(z,n); psi = -BE(n/2-1,z)/z; end
    %function psi = dfrhoeFD(z,n); Qm=-Polylog(n/2-1,-z); Qo=-Polylog( n/2 ,-z); Qp=-Polylog(n/2+1,-z); psi= ((2+n)/n) * (Qo^((2+n)/n-1)*Qm / (z*Qp)) - Qo^(2*(1+n)/n) / (z*Qp^2); end
    %function psi = dfrhoeBE(z,n); Qm= Polylog(n/2-1, z); Qo= Polylog( n/2 , z); Qp= Polylog(n/2+1, z); psi= ((2+n)/n) * (Qo^((2+n)/n-1)*Qm / (z*Qp)) - Qo^(2*(1+n)/n) / (z*Qp^2); end

end