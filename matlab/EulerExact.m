function [x, rho, ux, p, e, z, t, Mach, entro] = ...
    EulerExact(rho1, u1, t1, rho4, u4, t4, tEnd, n, h, statistic)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPDX-License-Identifier: MIT
% Copyright (c) 2014 Manuel A. Diaz
%
% Exact Riemann Solver for Classical and Quantum ideal gases
%
% This is a slight modification of the original code by 譚夢寧 (2015).
%
% Kinetic inputs (rho, u, T) are converted to effective pressures via
% fugacity before the classical solve; temperature and fugacity are
% corrected afterward for FD/BE statistics.
%
%   statistic = 'FD' | 'BE' | 'MB'  (Maxwell-Boltzmann reduces to classical gas)
%   gamma = (n+2)/n : adiabatic ratio
%   n : number of degrees of freedom
%   h : Planck length
%
%   Samples on x = 0:0.002:1 with discontinuity at x = 0.5.
%
% Example usage:
%   [x,rho,ux,p,e,z,t,Mach,entro] = EulerExact(...
%       rhoL,uL,tL,rhoR,uR,tR,tEnd,n,h,statistic)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref.:
% [1] 譚夢寧. (2015). 半古典波茲曼 BGK 方程式之任意統計稀薄流模擬.
%     臺灣大學應用力學研究所學位論文, 1-120.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Quantum Functions
BE = @(n,z)  PolyLog(n, z);
FD = @(n,z) -PolyLog(n,-z);

frhot = @(Q,z,n,rho,t,h) rho*h^n./(2*pi*t).^(n/2) - Q(n/2,z);
dfrhot = @(Q,z,n) -Q(n/2-1,z)/z;

frhoe = @(Q,z,n,rho,e,h) ((Q(n/2,z)).^((n+2)/n))./Q((n+2)/2,z) - (h^2)*n*rho.^(2/n)./(4*pi*e);
dfrhoe = @(Q,z,n) ((2+n)/n) .* (Q(n/2,z).^((2+n)/n-1).*Q(n/2-1,z) ./ (z.*Q(n/2+1,z))) - Q(n/2,z).^(2*(1+n)/n) ./ (z.*Q(n/2+1,z).^2);

% Define fugacity and Pressure of gas
switch statistic
    case 'FD' % Find z1, p1, z4 and p4 for FD gas
        z1=0.001; delta=1; %it=0;
        while abs(delta) > 1E-6
            z = z1 - frhot(FD,z1,n,rho1,t1,h) ./ dfrhot(FD,z1,n);
            delta=z-z1; z1=z; %disp(z)
            % it=it+1;
        end
        p1 = rho1*t1*FD(n/2+1,z1)/FD(n/2,z1); %disp(it);

        z4 = 0.001; delta = 1; %it = 0;
        while abs(delta) > 1E-6
            z = z4 - frhot(FD,z4,n,rho4,t4,h) ./ dfrhot(FD,z4,n);
            delta=z-z4; z4=z; %disp(z)
            % it=it+1;
        end
        p4 = rho4*t4*FD(n/2+1,z4)/FD(n/2,z4); %disp(it);

    case 'MB'
        p1 = rho1*t1;
        p4 = rho4*t4;

    case 'BE' % Find z1, p1, z4 and p4 for BE gas
        z1=0.001; delta=1; % it=0;
        while abs(delta) > 1E-6
            z = z1 - frhot(BE,z1,n,rho1,t1,h) ./ dfrhot(BE,z1,n);
            if z > 1; z=0.999999; end
            delta = z-z1; z1=z; %disp(z)
            % it=it+1;
        end
        p1 = rho1*t1*BE(n/2+1,z1)/BE(n/2,z1); %disp(it);

        z4=0.001; delta=1; % it=0;
        while abs(delta) > 1E-6
            z = z4 - frhot(BE,z4,n,rho4,t4,h) ./ dfrhot(BE,z4,n);
            if z > 1; z=0.999999; end
            delta = z-z4; z4=z; %disp(z)
            % it=it+1;
        end
        p4 = rho4*t4*BE(n/2+1,z4)/BE(n/2,z4); %disp(it);
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
x = 0:0.002:1;
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

switch statistic
    case 'FD' % Find z for FD model
        zFermi0 = 0.001; delta = 1; %it = 0;
        while max(abs(delta)) > 1E-6
            zFermi = zFermi0 - frhoe(FD,zFermi0,n,rho,e,h) ./ dfrhoe(FD,zFermi0,n);
            delta = zFermi - zFermi0; zFermi0 = zFermi;
            % it = it + 1;
        end
        z = zFermi; %disp(it);
        t = ((rho./(-PolyLog(n/2,-z))).^(2/n))*(h^2)/(2*pi);
    case 'MB'
        t = 2/n.*e;
        z = rho*h^n./(2*pi*t).^(n/2);
    case 'BE' % Find z for BE model
        zBose0 = 0.001; delta = 1; %it = 0;
        while abs(delta) > 1E-6
            zBose = zBose0 - frhoe(BE,zBose0,n,rho,e,h) ./ dfrhoe(BE,zBose0,n);
            zBose(zBose>1) = 0.999999;
            delta = zBose - zBose0; zBose0 = zBose;
            % it = it + 1;
        end
        z = zBose; %disp(it);
        t = ((rho./( PolyLog(n/2, z))).^(2/n))*(h^2)/(2*pi);
end
