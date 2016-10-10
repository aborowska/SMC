function [theta_sim, eps_bar, eps_sim, C_T] = SimSmooth(y_star, v, F_inv, K, L, par_KFS, RND)
% conditional simulation of disturbances    
    [n, S] = size(y_star);
    H = par_KFS.H;      % <-- MATRIX
    Z = par_KFS.Z;      % <-- SCALAR
    
    r = zeros(1,S);
    eps_bar = zeros(n,S);
    eps_sim = zeros(n,S);
    theta_sim = zeros(n,S);
    N = zeros(1,S);
    
%     RND = randn(n,S);
    for ii = n:-1:1
        C = H(ii,:) - H(ii,:).*(F_inv(ii,:) + N.*K(ii,:).^2).*H(ii,:);
        eps_bar(ii,:) = H(ii,:).*(F_inv(ii,:).*v(ii,:) - K(ii,:).*r);
        w = (C.^(0.5)).*RND(ii,:);
        eps_sim(ii,:) = eps_bar(ii,:) + w; 
        theta_sim(ii,:) = y_star(ii,:) - eps_sim(ii,:);
        
        W = H(ii,:).*(F_inv(ii,:).*Z - K(ii,:).*N.*L(ii,:));
        r = Z.*F_inv(ii,:).*v(ii,:) - W.*w./C + L(ii,:).*r;
        N = Z.*F_inv(ii,:).*Z + W.*W./C + N.*(L(ii,:).^2);
    end
        
    C_T = H(n,:) - H(n,:).*F_inv(n,:).*H(n,:);  % Variance for the last state
    eps_bar = y_star(n,:) - eps_bar(n,:);       % Mean for the last state
    eps_sim = eps_sim(n,:);
end