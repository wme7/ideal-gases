%% Find z value for BE and FD gas result
% coded by Manuel Diaz, NTU, 2014.12.12.

clear; % clear memory

% From classical conditions
%example = [0.4,0.0,0.24 ]; % degenerate limit
%example = [2.0,0.5,0.315]; % degenerate limit
%example = [1.0,0.0,1.0  ];	% classical limit
%example = [0.125,0.5,0.1];	% classical limit
example = [0.7,0.0,1.0];	% classical limit
rho = example(1); u = example(2); p = example(3);

% temperature
t = p/rho;

% Degrees of Freedom
n = 3;

% Gamma and internal energy
gamma = (n+2)/n; e = p/((gamma-1)*rho);

% Planck constant
h = 6;

% find z through energy and density relation
%ferhoFD = @(z) (FD(n/2,z)).^((n+2)/n)./FD((n+2)/2,z)-(h^2)*n*rho^(2/n)/(4*pi*e);
%ferhoBE = @(z) (FD(n/2,z)).^((n+2)/n)./BE((n+2)/2,z)-(h^2)*n*rho^(2/n)/(4*pi*e);

% find z through density and temperature relation
%frhotFD = @(z) rho*h^n./(2*pi*t).^(n/2)-FD(n/2,z);
%frhotBE = @(z) rho*h^n./(2*pi*t).^(n/2)-BE(n/2,z);

% Notice that Polylog is faster than PolyLog for single value.

% Plot functions (I need to see that there is a only one root!)
% subplot(2,1,1); x=0.0001:0.01:1.0001; plot(x,ZBE(x)); hold on; 
%     line([0 1 ],[0,0],'color','k'); hold off; xlim([0,1]); 
%     title('Bose-Einstein'); ylabel('zBE(z)'); xlabel('z');
% subplot(2,1,2); x=0.0001:0.1:10.0001; plot(x,ZFD(x)); hold on; 
%     line([0 10],[0,0],'color','k'); hold off; xlim([0,10]); 
%     title('Fermi-Dirac'); ylabel('zFD(z)'); xlabel('z');

% Using Matlab built in functions
%tic
% Find z (degenerate limit)
%z = fzero(frhotBE, 1); disp(['zBE: ',num2str(z)]); 
%z = fzero(frhotFD,50); disp(['zFD: ',num2str(z)]);

% Find z (classical limit)
%z = fzero(frhotBE,0.05); disp(['zBE: ',num2str(z)]);
%z = fzero(frhotFD,0.05); disp(['zFD: ',num2str(z)]);
%toc

% Plank constants
%hFD = 6.0; 
%hBE = 3.3; 
hFD = 2.5066;
hBE = 2.5066;

% Using my costum Newton root finding method
tic
z0 = 0.001; delta = 1; it = 0;
while abs(delta) > 1E-6
    z=z0+frhotFD(z0,n,rho,t,hFD)/dfrhotFD(z0,n); 
    delta=z-z0; z0=z; %disp(z)
    it=it+1;
end
disp([z,it])

z0 = 0.001; delta = 1; it = 0;
while abs(delta) > 1E-6
    z=z0-frhotBE(z0,n,rho,t,hBE)/dfrhotBE(z0,n); if z > 1; z=0.999; end
    delta = z-z0; z0 = z; %disp(z)
    it=it+1;
end
disp([z,it])
toc

% NOTE: 
% BE function is a monotonically increasing function!
% FD function is a monotonically decreasing function!

%% if internal energy is an initial condition

%example = [1.0,0.0,2/3];
example = [0.7,0.0,1.0];
rho = example(1); u = example(2); p = example(3);

n=3; x=1; type = 'BE'; h=2.5066;

gamma=(n+2)/n;          % gamma constant
e = p./((gamma-1).*rho);	% internal energy

switch type
    case 'FD' % Find z for FD model
        for i = 1:length(x)
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
        for i = 1:length(x)
            zBose0 = 0.001; delta = 1;
            while abs(delta) > 1E-6
                zBose=zBose0-frhoeBE(zBose0,n,rho(i),e(i),h)/dfrhoeBE(zBose0,n); if zBose>1; zBose=0.999; end
                delta=zBose-zBose0; zBose0=zBose;
            end
            z(i) = zBose;   
        end
        t = ((rho./( PolyLog(n/2, z))).^(2/n))*(h^2)/(2*pi);
end

% therefore the initial condition in ShiYang2008 should be:
    r0 = [1.0,0.7]; ux0=[0,0]; e0=[1.0,1.5]; t0=[0.8617,1.1183];
    
% still no good ... I suspect is because of the normalization.
