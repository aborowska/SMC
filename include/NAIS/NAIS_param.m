function [par_NAIS, theta_smooth]= NAIS_param(par_NAIS, y, par_SV, cont)
    % Algorithm 2: Efficient importance parameters using NAIS
    n = length(y);
    Un = ones(n,1);
      
    tol = cont.tol;     % convergence tolerance
    tol_C = cont.tol_C;
    z = cont.GH.z;      % the Gauss-Hermite abscissae z 
    h = cont.GH.h;      % and the associated weights h(z)
    w = sqrt(h);
    iter_max = cont.iter_max;                    

    err = 1;
    iter = 0;
    
    if (cont.err == 't')
        nu = par_SV(1,4);
        pdf_const = log(gamma((nu+1)/2)) - log(gamma(nu/2)) - 0.5*log(nu-2);
%         pdf_const = 0;
    end
            
    while ((err > tol) && (iter < iter_max))        
        par_KFS = IS_Model(par_NAIS, par_SV);
        b = par_NAIS.b;
        C = par_NAIS.C;
        y_star = b./C;

%         [theta_smooth, V_smooth] = KFS(y_star, par_KFS); 
%          [theta_smooth, V_smooth, ~, F_inv, epsilon, K, L]  = KFS(y_star, par_KFS); 
        [theta_smooth, V_smooth] = KFS_mex(y_star, par_KFS.P1, par_KFS.c, ...
                                   par_KFS.H, par_KFS.Q, par_KFS.d, par_KFS.T, par_KFS.R, par_KFS.Z);

        
        % compute the smoothed mean theta_smooth
        % and smoothed variance V_smooth based on b, C from previous
        % iteration and the linear SSM using KFS

%         b_new = 0*Un;
%         C_new = 1*Un;
%          
%         for ii = 1:n
%             % generate the nodes of GH integration
%             theta_GH = theta_smooth(ii,1) + sqrt(V_smooth(ii,1))*z;
%             % weighted least squares regression
%             if (cont.err == 'n')
% %                 Y = -0.5*(theta_GH + (y(ii,1)^2)./exp(theta_GH)); 
%                 Y = -0.5*(log(2*pi) + theta_GH + (y(ii,1)^2)./exp(theta_GH)); 
%             else % if cont.err == 't'
%                 Y = pdf_const - 0.5*(theta_GH + (nu+1)*log(1 + (y(ii,1)^2)./((nu-2).*exp(theta_GH)))); 
% %                 y2 = (y(ii,1)^2)./((nu-2)*exp(theta_GH));
% %                 Y =  - 0.5*theta_GH - 0.5*(nu+1)*log(1 + y2); 
%             end
% %             X = [theta_GH, -0.5*theta_GH.^2];
% %             X = repmat(w,1,2).*X;
% %             X = [w, X];
% 
%             w_theta_GH = w.*theta_GH;
%             w_theta_GH2 = -0.5*w.*theta_GH.^2;
%             X = [w, w_theta_GH, w_theta_GH2];
%             XT = X';
%             Y = w.*Y;
%             beta = (XT*X)\(XT*Y);
%             b_new(ii,1) = beta(2,1);
%             if (beta(3,1) < tol_C)
%                 C_new(ii,1) = tol_C;
%             else 
%                 C_new(ii,1) = beta(3,1);        
%             end
%             
% %             beta = EIS_reg(Y,theta_GH,w);
% %             b_new(ii,1) = beta(1,1);
% %             if (beta(2,1) < tol_C)
% %                 C_new(ii,1) = tol_C;
% %             else 
% %                 C_new(ii,1) = beta(2,1);        
% %             end
%         end

        if (cont.err == 'n')
            [b_new, C_new] = EIS_reg_vec(y, theta_smooth, V_smooth, z, w, tol_C);
        else % if cont.err == 't'
            [b_new, C_new] = EIS_reg_vec_t(y, theta_smooth, V_smooth, z, w, tol_C, nu, pdf_const);
        end
        
        err_b = sum((par_NAIS.b - b_new).^2)/n;
        err_C = sum((par_NAIS.C - C_new).^2)/n;
        err = max(err_b, err_C);
        
        par_NAIS.b = b_new;
        par_NAIS.C = C_new;
        iter = iter + 1;
    end
%     fprintf('NAIS_param iter #: %d.\n', iter)
end