function [x, rho, ux, p, e, z, t, Mach, entro] = ...
    QEulerExactToro(rhoL, uL, tL, rhoR, uR, tR, tEnd, n, h, statistic)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPDX-License-Identifier: MIT
% Copyright (c) 2014 Manuel A. Diaz
%
% Exact Riemann Solver for Classical and Quantum ideal gases
%
% This is an extension of the original code by 譚夢寧 (2015) using Toro's 
% Exact Riemann solver (Toro, 1999).
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
%   [x,rho,ux,p,e,z,t,Mach,entro] = QEulerExactToro(...
%       rhoL,uL,tL,rhoR,uR,tR,tEnd,n,h,statistic)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Refs.:
% [1] 譚夢寧. (2015). 半古典波茲曼 BGK 方程式之任意統計稀薄流模擬. 
%     臺灣大學應用力學研究所學位論文, 1-120.
% [2] Toro, E. F. (1999). Riemann solvers and numerical methods for fluid 
%     dynamics. Springer Science & Business Media.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    gamma = (n + 2) / n;
    x0 = 0.5;
    x = 0:0.002:1;
    pFloor = 1e-300;
    rhoFloor = 1e-10;
        
    % --- Quantum Functions ---
    BE = @(n,z)  PolyLog(n, z);
    FD = @(n,z) -PolyLog(n,-z);

    % --- Pre-process: kinetic ICs -> effective pressures for Toro solver ---
    switch statistic
        case 'FD'
            if rhoL > rhoFloor
                zL = 0.001; delta = 1;
                while abs(delta) > 1e-6
                    z = zL - frhot(FD, zL, n, rhoL, tL, h) ./ dfrhot(FD, zL, n);
                    delta = z - zL; zL = z;
                end
                pL = rhoL * tL * FD(n / 2 + 1, zL) / FD(n / 2, zL);
            else
                pL = 0;
            end

            if rhoR > rhoFloor
                zR = 0.001; delta = 1;
                while abs(delta) > 1e-6
                    z = zR - frhot(FD, zR, n, rhoR, tR, h) ./ dfrhot(FD, zR, n);
                    delta = z - zR; zR = z;
                end
                pR = rhoR * tR * FD(n / 2 + 1, zR) / FD(n / 2, zR);
            else
                pR = 0;
            end

        case 'MB'
            pL = rhoL * tL;
            pR = rhoR * tR;

        case 'BE'
            if rhoL > rhoFloor
                zL = 0.001; delta = 1;
                while abs(delta) > 1e-6
                    z = zL - frhot(BE, zL, n, rhoL, tL, h) ./ dfrhot(BE, zL, n);
                    if z > 1; z = 0.999999; end % z : [0,1[
                    delta = z - zL; zL = z;
                end
                pL = rhoL * tL * BE(n / 2 + 1, zL) / BE(n / 2, zL);
            else
                pL = 0;
            end

            if rhoR > rhoFloor
                zR = 0.001; delta = 1;
                while abs(delta) > 1e-6
                    z = zR - frhot(BE, zR, n, rhoR, tR, h) ./ dfrhot(BE, zR, n);
                    if z > 1; z = 0.999999; end % z : [0,1[
                    delta = z - zR; zR = z;
                end
                pR = rhoR * tR * BE(n / 2 + 1, zR) / BE(n / 2, zR);
            else
                pR = 0;
            end

        otherwise
            error('QEulerExactToro:Unknownstatistic', ...
                'Unknown quantum gas statistic "%s".', statistic);
    end

    % --- Toro Riemann solve in effective pressure space ---
    sol = riemannExactState(rhoL, uL, pL, rhoR, uR, pR, gamma);
    [rho, ux, p] = sampleRiemannProfile(sol, x, x0, tEnd, gamma);

    % --- Classical derived quantities ---
    rho = max(rho, rhoFloor);
    p = max(p, pFloor);
    c = sqrt(max(gamma * p ./ rho, 0));
    Mach = ux ./ max(c, 1e-12);
    entro = log(p ./ rho.^gamma);
    e = p ./ ((gamma - 1) * rho);

    % --- Post-process: quantum fugacity and temperature correction ---
    switch statistic
        case 'FD'
            zFermi0 = 0.001; delta = 1;
            while abs(delta) > 1e-6
                zFermi = zFermi0 - frhoe(FD, zFermi0, n, rho, e, h) ./ dfrhoe(FD, zFermi0, n);
                delta = zFermi - zFermi0; zFermi0 = zFermi;
            end
            z = zFermi;
            t = ((rho ./ (FD(n / 2, z))).^(2 / n)) * (h^2) / (2 * pi);

        case 'MB'
            t = (2 / n) .* e;
            z = rho * h^n ./ (2 * pi * t).^(n / 2);

        case 'BE'
            zBose0 = 0.001; delta = 1;
            while abs(delta) > 1e-6
                zBose = zBose0 - frhoe(BE, zBose0, n, rho, e, h) ./ dfrhoe(BE, zBose0, n);
                zBose(zBose > 1) = 0.999999; % z : [0,1[
                delta = zBose - zBose0; zBose0 = zBose;
            end
            z = zBose;
            t = ((rho ./ BE(n / 2, z)).^(2 / n)) * (h^2) / (2 * pi);
    end
end

% =========================================================================
function sol = riemannExactState(rhoL, uL, pL, rhoR, uR, pR, gamma)
%RIEMANNEXACTSTATE  Star state and wave statistics (Toro Algorithm 4.1 + vacuum).

    cL = sqrt(gamma * pL / max(rhoL, 1e-300));
    cR = sqrt(gamma * pR / max(rhoR, 1e-300));

    dryVelL = uL + 2 * cL / (gamma - 1);
    dryVelR = uR - 2 * cR / (gamma - 1);

    sol.gamma = gamma;
    sol.rhoL = rhoL; sol.uL = uL; sol.pL = pL; sol.cL = cL;
    sol.rhoR = rhoR; sol.uR = uR; sol.pR = pR; sol.cR = cR;

    if pL < 1e-10 && rhoL < 1e-10
        sol.caseId = 'vacuumLeft';
        return;
    end
    if pR < 1e-10 && rhoR < 1e-10
        sol.caseId = 'vacuumRight';
        return;
    end
    if dryVelL <= dryVelR
        sol.caseId = 'vacuumMiddle';
        sol.dryVelL = dryVelL;
        sol.dryVelR = dryVelR;
        return;
    end

    pStar = max(1e-8, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (cL + cR));
    change = 1.0;
    tol = 1e-6;
    maxIter = 50;

    for iter = 1:maxIter
        [fL, dfL] = pressureWave(pStar, pL, rhoL, gamma);
        [fR, dfR] = pressureWave(pStar, pR, rhoR, gamma);
        f = fL + fR + uR - uL;
        change = abs(f);
        if change < tol
            break;
        end
        pStar = max(1e-10, pStar - f / (dfL + dfR));
    end

    if iter == maxIter && change > 1e-3
        error('QEulerExactToro:StarPressure', ...
            'Newton iteration did not converge (|f| = %.3e).', change);
    end

    [~, ~, waveL] = pressureWave(pStar, pL, rhoL, gamma);
    [~, ~, waveR] = pressureWave(pStar, pR, rhoR, gamma);

    uStar = uL - pressureWaveValue(pStar, pL, rhoL, gamma);
    rhoLStar = starDensity(pStar, pL, rhoL, gamma, waveL);
    rhoRStar = starDensity(pStar, pR, rhoR, gamma, waveR);

    sol.caseId = 'standard';
    sol.pStar = pStar;
    sol.uStar = uStar;
    sol.rhoLStar = rhoLStar;
    sol.rhoRStar = rhoRStar;
    sol.waveL = waveL;
    sol.waveR = waveR;
end

% =========================================================================
function [rho, ux, p] = sampleRiemannProfile(sol, x, x0, t, gamma)
%SAMPLERIEMANNPROFILE  Pointwise exact solution at time t.

    rho = zeros(size(x));
    ux  = zeros(size(x));
    p   = zeros(size(x));

    if t <= 0
        for i = 1:numel(x)
            if x(i) < x0
                rho(i) = sol.rhoL; ux(i) = sol.uL; p(i) = sol.pL;
            else
                rho(i) = sol.rhoR; ux(i) = sol.uR; p(i) = sol.pR;
            end
        end
        return;
    end

    for i = 1:numel(x)
        [rho(i), ux(i), p(i)] = sampleAtPoint(sol, x(i), x0, t, gamma);
    end
end

% =========================================================================
function [rho, u, pOut] = sampleAtPoint(sol, xi, x0, t, gamma)
%SAMPLEATPOINT  Exact primitive state at a single (x,t) location.

    switch sol.caseId
        case 'vacuumLeft'
            if xi < x0
                rho = 0; u = 0; pOut = 0;
            else
                rho = sol.rhoR; u = sol.uR; pOut = sol.pR;
            end
            return;

        case 'vacuumRight'
            if xi < x0
                rho = sol.rhoL; u = sol.uL; pOut = sol.pL;
            else
                rho = 0; u = 0; pOut = 0;
            end
            return;

        case 'vacuumMiddle'
            [rho, u, pOut] = sampleVacuumMiddlePoint(sol, xi, x0, t, gamma);
            return;

        case 'standard'
            [rho, u, pOut] = sampleStandardPoint(sol, xi, x0, t, gamma);
            return;

        otherwise
            error('QEulerExactToro:UnknownCase', 'Unknown Riemann case "%s".', sol.caseId);
    end
end

% =========================================================================
function [rho, u, pOut] = sampleStandardPoint(sol, xi, x0, t, gamma)
%SAMPLESTANDARDPOINT  Standard four-wave pattern (shocks and/or rarefactions).

    s = (xi - x0) / t;
    rhoL = sol.rhoL; uL = sol.uL; pL = sol.pL; cL = sol.cL;
    rhoR = sol.rhoR; uR = sol.uR; pR = sol.pR; cR = sol.cR;
    pStar = sol.pStar; uStar = sol.uStar;
    rhoLStar = sol.rhoLStar; rhoRStar = sol.rhoRStar;

    leftActive  = abs(rhoLStar - rhoL) > 1e-8 * max(rhoL, 1) || abs(uStar - uL) > 1e-8;
    rightActive = abs(rhoRStar - rhoR) > 1e-8 * max(rhoR, 1) || abs(uStar - uR) > 1e-8;

    if leftActive
        if strcmp(sol.waveL, 'shock')
            leftOuter = shockSpeed(rhoL, uL, rhoLStar, uStar);
        else
            cLStar = sqrt(gamma * pStar / rhoLStar);
            leftOuter = uL - cL;
            leftInner = uStar - cLStar;
        end
    end

    if rightActive
        if strcmp(sol.waveR, 'shock')
            rightOuter = shockSpeed(rhoR, uR, rhoRStar, uStar);
        else
            cRStar = sqrt(gamma * pStar / rhoRStar);
            rightInner = uStar + cRStar;
            rightOuter = uR + cR;
        end
    end

    if leftActive
        if strcmp(sol.waveL, 'shock')
            if s <= leftOuter
                rho = rhoL; u = uL; pOut = pL;
                return;
            end
        else
            if s <= leftOuter
                rho = rhoL; u = uL; pOut = pL;
                return;
            elseif s <= leftInner
                [u, pOut, rho] = rarefactionStateLeft(s, gamma, rhoL, uL, pL, cL);
                return;
            end
        end
    end

    if s <= uStar
        rho = rhoLStar; u = uStar; pOut = pStar;
        return;
    end

    if rightActive
        if strcmp(sol.waveR, 'shock')
            if s <= rightOuter
                rho = rhoRStar; u = uStar; pOut = pStar;
                return;
            end
        else
            if s <= rightInner
                rho = rhoRStar; u = uStar; pOut = pStar;
                return;
            elseif s <= rightOuter
                [u, pOut, rho] = rarefactionStateRight(s, gamma, rhoR, uR, pR, cR);
                return;
            end
        end
    end

    rho = rhoR; u = uR; pOut = pR;
end

% =========================================================================
function [rho, u, pOut] = sampleVacuumMiddlePoint(sol, xi, x0, t, gamma)
%SAMPLEVACUUMMIDDLEPOINT  Toro section 4.1.3 — vacuum region between rarefactions.

    s = (xi - x0) / t;
    rhoL = sol.rhoL; uL = sol.uL; pL = sol.pL; cL = sol.cL;
    rhoR = sol.rhoR; uR = sol.uR; pR = sol.pR; cR = sol.cR;
    dryVelL = sol.dryVelL;
    dryVelR = sol.dryVelR;

    if s <= uL - cL
        rho = rhoL; u = uL; pOut = pL;
    elseif s <= dryVelL
        [u, pOut, rho] = rarefactionStateLeft(s, gamma, rhoL, uL, pL, cL);
    elseif s <= dryVelR
        rho = 0; u = 0; pOut = 0;
    elseif s <= uR + cR
        [u, pOut, rho] = rarefactionStateRight(s, gamma, rhoR, uR, pR, cR);
    else
        rho = rhoR; u = uR; pOut = pR;
    end
end

% =========================================================================
function [u, pOut, rho] = rarefactionStateLeft(s, gamma, rhoK, uK, pK, cK)
%RAREFACTIONSTATELEFT  Left rarefaction fan (general u_K, Toro section 4.3).

    u = ((gamma - 1) * uK + 2 * cK) / (gamma + 1) + (2 / (gamma + 1)) * s;
    c = (uK + 2 * cK / (gamma - 1) - u) * (gamma - 1) / 2;
    cRatio = max(c / cK, 0);
    pOut = pK * cRatio^(2 * gamma / (gamma - 1));
    rho = rhoK * cRatio^(2 / (gamma - 1));
end

% =========================================================================
function [u, pOut, rho] = rarefactionStateRight(s, gamma, rhoK, uK, pK, cK)
%RAREFACTIONSTATERIGHT  Right rarefaction fan (general u_K).

    u = ((gamma - 1) * uK - 2 * cK) / (gamma + 1) + (2 / (gamma + 1)) * s;
    c = (u + 2 * cK / (gamma - 1) - uK) * (gamma - 1) / 2;
    cRatio = max(c / cK, 0);
    pOut = pK * cRatio^(2 * gamma / (gamma - 1));
    rho = rhoK * cRatio^(2 / (gamma - 1));
end

% =========================================================================
function [f, df, wavestatistic] = pressureWave(p, pK, rhoK, gamma)
%PRESSUREWAVE  Toro f_K(p) and derivative (eqs. 4.8, 4.42, 4.43).

    cK = sqrt(gamma * pK / rhoK);
    if p >= pK
        wavestatistic = 'shock';
        A = 2 / ((gamma + 1) * rhoK);
        B = (gamma - 1) / (gamma + 1) * pK;
        f = (p - pK) * sqrt(A / (p + B));
        df = sqrt(A / (p + B)) * (1 - 0.5 * (p - pK) / (p + B));
    else
        wavestatistic = 'rarefaction';
        pr = p / pK;
        f = 2 * cK / (gamma - 1) * (pr^((gamma - 1) / (2 * gamma)) - 1);
        df = (1 / (rhoK * cK)) * pr^(-(gamma + 1) / (2 * gamma));
    end
end

% =========================================================================
function f = pressureWaveValue(p, pK, rhoK, gamma)
    [f, ~, ~] = pressureWave(p, pK, rhoK, gamma);
end

% =========================================================================
function rhoStar = starDensity(pStar, pK, rhoK, gamma, wavestatistic)
    if strcmp(wavestatistic, 'shock')
        rhoStar = rhoK * ((gamma + 1) * pStar + (gamma - 1) * pK) / ...
            ((gamma - 1) * pStar + (gamma + 1) * pK);
    else
        rhoStar = rhoK * (pStar / pK)^(1 / gamma);
    end
end

% =========================================================================
function s = shockSpeed(rhoK, uK, rhoStar, uStar)
%SHOCKSPEED  Rankine-Hugoniot shock speed (Toro eq. 4.70).

    s = (rhoK * uK - rhoStar * uStar) / (rhoK - rhoStar);
end

% =========================================================================
% Quantum EOS helpers
% =========================================================================
function psi = frhot(Q,z,n,rho,t,h)
    psi = rho*h^n./(2*pi*t).^(n/2) - Q(n/2,z);
end

function dpsi = dfrhot(Q,z,n)
    dpsi = -Q(n/2-1,z)./z;
end

function psi = frhoe(Q,z,n,rho,e,h)
    psi = ((Q(n/2,z)).^((n+2)/n)) ./ Q((n+2)/2,z) - (h^2) * n * rho.^(2/n)./(4 * pi * e);
end

function dpsi = dfrhoe(Q,z,n)
    Qm = Q(n/2-1,z);
    Qo = Q(n/2,z);
    Qp = Q(n/2+1,z);
    dpsi = ((2+n)/n) .* (Qo.^((2+n)/n-1) .* Qm ./ (z.*Qp)) - Qo.^(2*(1+n)/n) ./ (z.*Qp.^2);
end