function [theta_smooth, V, nu, F_inv, epsilon, K, L]  = KFS(y,param)
% "parallel" univariate Kalman Filter and Smoother 
% for multiple parameter vectors
% n = size(y,1); 
    [n,S] = size(param.H);
    
%% Set parameters and initial values
    a = zeros(n+1,S);     % filtered state
    P = zeros(n+1,S);     % filtered state variance
    
    P(1,:) = param.P1;  % <-- VECTOR
    c = param.c;        % <-- VECTOR
    H = param.H;    	% <-- MATRIX
    Q = param.Q;        % <-- VECTOR
    d = param.d;        % <-- SCALAR
    T = param.T;        % <-- VECTOR
    R = param.R;        % <-- SCALAR
    Z = param.Z;        % <-- SCALAR
    
%% Output of Kalman filter
    nu = zeros(n,S);            % prediction errors
    F = zeros(n,S);             % prediction variance
    F_inv = zeros(n,S);
    K = zeros(n,S);             % = P/F, Kalman gain (regression coefficient of alpha on nu)
    L = zeros(n,S);  
    for ii = 1:n
        nu(ii,:) = y(ii,:) - c(1,:) - Z.*a(ii,:);
        F(ii,:) = Z.*P(ii,:).*Z + H(ii,:);
        F_inv(ii,:) = 1./F(ii,:);
        K(ii,:) = T(1,:).*P(ii,:).*Z./F(ii,:);
        L(ii,:) = T(1,:) - K(ii,:).*Z;
        a(ii+1,:) = d + T(1,:).*a(ii,:) + K(ii,:).*nu(ii,:);
        P(ii+1,:) = T(1,:).*P(ii,:).*T(1,:) + R.*Q(1,:).*R - K(ii,:).*F(ii,:).*K(ii,:);     
    end
    
%% State smoothing 
%     alpha_smooth = zeros(n,1);  % smoothed state
%     V = zeros(n,1);             % smoothed state variance
%     r = zeros(n,1);             % smoothing cumulant
%     N = zeros(n,1);             % smoothing variance cumulant
% 
%     for ii = n:-1:2
%         r(ii-1,1) = nu(ii,1)/F(ii,1) + L(ii,1)*r(ii,1);
%         alpha_smooth(ii,1) = a(ii,1) + P(ii,1)*r(ii-1,1);
%         N(ii-1,1) = 1/F(ii,1) + L(ii,1)*N(ii,1)*L(ii,1);
%         V(ii,1) = P(ii,1) - P(ii,1)*N(ii-1,1)*P(ii,1);
%     end   
%     theta_smooth = c + Z.*alpha_smooth;
    
%% Distrurbance smoothing   %         [epsilon,V]=libdistsmo(Z,H,T||v,FINV,K,L,[]);
    epsilon = zeros(n,S);       % smoothed disturbance
    theta_smooth = zeros(n,S);       % smoothed signal
    C = zeros(n,S);             % smoothed disturbance variance
    N = zeros(1,S);
    r = zeros(1,S);
    
    for ii = n:-1:1
        D = F_inv(ii,:) + K(ii,:).*N.*K(ii,:);
        C(ii,:) = H(ii,:) - H(ii,:).*D.*H(ii,:);
        u = nu(ii,:).*F_inv(ii,:) - K(ii,:).*r;
        epsilon(ii,:) = H(ii,:).*u;
        theta_smooth(ii,:) = y(ii,:) - epsilon(ii,:);
        r = Z.*nu(ii,:).*F_inv(ii,:) + L(ii,:).*r;
        N = Z.*Z.*F_inv(ii,:) + L(ii,:).*N.*L(ii,:);
    end
    
%% Recovering the smoothed signal: alpha_smooth = y-epsilon;
%     theta_smooth = kron(y,ones(1,S)) - epsilon;
    V = C;
end