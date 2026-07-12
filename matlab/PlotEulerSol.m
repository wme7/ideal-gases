%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Plot Quantum Euler Solutions in classical and quantum regimes
%
%               Coded by Manuel Diaz, NTU, 2014.12.23.
%               SPDX-License-Identifier: MIT
%               Copyright (c) 2014 Manuel A. Diaz
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear memory and close figures
clear; clc; close all;

%% setup matlab enviroment

% set(0,'defaultTextInterpreter','none') 
set(0,'defaultTextInterpreter','latex')
set(0,'DefaultTextFontName','Times',...
'DefaultTextFontSize',14,...
'DefaultAxesFontName','Times',...
'DefaultAxesFontSize',8,...
'DefaultLineLineWidth',1.5,...
'DefaultAxesBox','on',...
'defaultAxesLineWidth',1.0,...
'DefaultFigureColor','w',...
'DefaultLineMarkerSize',7.75)

% list of options for lines
color=['k','r','m','b','c','g','y','w'];
lines={'-',':','--','-.','-.','none'};
mark=['s','+','o','x','v','none'];

% local marker size
ms = 6;

% save to path
path = './';

%% Plot selected example
example = 5;

switch example
    case 1 % Sod's shock-tube problem
        n = 5;      % degress of freedom (DoF)
        h = 1.0;    % plank constant
        tEnd=0.1;   % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExact(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExact(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExact(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'FD');
	case 2 % Sod's shock-tube problem
        n = 5;      % degress of freedom (DoF)
        h = 2;      % plank constant
        tEnd=0.1;   % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExact(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExact(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExact(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'FD');
	case 3 % Sod's shock-tube problem
        n = 5;      % degress of freedom (DoF)
        h = 2.65;	% plank constant
        tEnd=0.1;   % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExactToro(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExactToro(1,0,1,0.125,0,0.1/0.125,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExactToro(1,0,1,0.125,0,0.1/0.125,tEnd,n,6.0,'FD');
    case 4 % Hu and Jin Problems in classical regime
        n = 3;      % degress of freedom (DoF)
        h = 1.0;    % plank constant
        tEnd=0.18;  % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExact(1,0,1,0.4,0,0.6,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExact(1,0,1,0.4,0,0.6,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExact(1,0,1,0.4,0,0.6,tEnd,n,h,'FD');
    case 5 % Hu and Jin Problems in Degenerate regime
        n = 3;      % degress of freedom (DoF)
        h = 3.3;    % plank constant
        tEnd=0.18;  % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExact(1,0,1,0.4,0,0.6,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExact(1,0,1,0.4,0,0.6,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExact(1,0,1,0.4,0,0.6,tEnd,n,6.0,'FD');
    case 6 % Yang and Shi Degenerate regime problem.
        n = 3;    	% degress of freedom (DoF)
        h = 6.0;      % plank constant
        tEnd=0.10;  % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExact(3.086455,0,8.053324,3.084272,0,8.067390,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExact(3.086455,0,8.053324,3.084272,0,8.067390,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExact(3.086455,0,8.053324,3.084272,0,8.067390,tEnd,n,h,'FD');
    case 7 % Filbet and Jing (2010).
        n = 2;    	% degress of freedom (DoF)
        h = 0.1;    % plank constant
        tEnd=0.20;  % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExactToro(1,0,1,0.125,0,0.25,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExactToro(1,0,1,0.125,0,0.25,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExactToro(1,0,1,0.125,0,0.25,tEnd,n,h,'FD');
	case 8 % Filbet and Jing (2010).
        n = 2;    	% degress of freedom (DoF)
        h = 3.0;    % plank constant
        tEnd=0.20;  % output time
        [~,rhoBE,uxBE,pBE,eBE,zBE,tBE]=EulerExactToro(1,0,1,0.125,0,0.25,tEnd,n,h,'BE');
        [~,rhoMB,uxMB,pMB,eMB,zMB,tMB]=EulerExactToro(1,0,1,0.125,0,0.25,tEnd,n,h,'MB');
        [x,rhoFD,uxFD,pFD,eFD,zFD,tFD]=EulerExactToro(1,0,1,0.125,0,0.25,tEnd,n,h,'FD');
end

%% Plot Individial pictures
figure(1)

subplot(3,6,1); plot(x,rhoFD,color(3)); title('$\rho$-FD'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,7); plot(x,rhoMB,color(4)); title('$\rho$-MB'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,13); plot(x,rhoBE,color(2)); title('$\rho$-BE'); axis('square','tight'); xlabel('$x$'); 

subplot(3,6,2); plot(x,uxFD,color(3)); title('$u_x$-FD'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,8); plot(x,uxMB,color(4)); title('$u_x$-MB'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,14); plot(x,uxBE,color(2)); title('$u_x$-BE'); axis('square','tight'); xlabel('$x$'); 

subplot(3,6,3); plot(x,pFD,color(3)); title('$p$-FD'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,9); plot(x,pMB,color(4)); title('$p$-MB'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,15); plot(x,pBE,color(2)); title('$p$-BE'); axis('square','tight'); xlabel('$x$'); 

subplot(3,6,4); plot(x,eFD,color(3)); title('$e$-FD'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,10); plot(x,eMB,color(4)); title('$e$-MB'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,16); plot(x,eBE,color(2)); title('$e$-BE'); axis('square','tight'); xlabel('$x$'); 

subplot(3,6,5); plot(x,tFD,color(3)); title('$\theta$-FD'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,11); plot(x,tMB,color(4)); title('$\theta$-MB'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,17); plot(x,tBE,color(2)); title('$\theta$-BE'); axis('square','tight'); xlabel('$x$'); 

subplot(3,6,6); plot(x,zFD,color(3)); title('$z$-FD'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,12); plot(x,zMB,color(4)); title('$z$-MB'); axis('square','tight'); %xlabel('$x$'); 
subplot(3,6,18); plot(x,zBE,color(2)); title('$z$-BE'); axis('square','tight'); xlabel('$x$'); 

% Export plot to *.eps figure.
print('-dpng',[path,'QEuler_Eg',num2str(example),'_AllPlots.png']);

%% compute aparent gamma value
%gammaBE = (n+2)/n*( PolyLog(n/2+1, zFD)).*( PolyLog(n/2-1, zFD))./( PolyLog(n/2, zFD)).^2;
%gammaMB = (n+2)/n;
%gammaFD = (n+2)/n*(-PolyLog(n/2+1,-zBE)).*(-PolyLog(n/2-1,-zBE))./(-PolyLog(n/2,-zBE)).^2;

%% Plot Comparative figures
figure(2)
statistic = {'BE', 'MB', 'FD'}; % place outside on top
subplot(3,2,1); plot(x,rhoBE,color(2),x,rhoMB,color(4),x,rhoFD,color(3)); ylabel('$\rho$'); xlabel('$x$'); legend(statistic,'Location','best');
subplot(3,2,2); plot(x,uxBE,color(2),x,uxMB,color(4),x,uxFD,color(3)); ylabel('$u_x$'); xlabel('$x$'); legend(statistic,'Location','best');
subplot(3,2,4); plot(x,pBE,color(2),x,pMB,color(4),x,pFD,color(3)); ylabel('$p$'); xlabel('$x$'); legend(statistic,'Location','best');
subplot(3,2,3); plot(x,eBE,color(2),x,eMB,color(4),x,eFD,color(3)); ylabel('$e$'); xlabel('$x$'); legend(statistic,'Location','best');
subplot(3,2,5); plot(x,zBE,color(2),x,zMB,color(4),x,zFD,color(3)); ylabel('$z$'); xlabel('$x$'); legend(statistic,'Location','best');
subplot(3,2,6); plot(x,tBE,color(2),x,tMB,color(4),x,tFD,color(3)); ylabel('$\theta$'); xlabel('$x$'); legend(statistic,'Location','best');

% Export plot to *.eps figure.
print('-dpng',[path,'QEuler_Eg',num2str(example),'_TogetherPlots.png']);