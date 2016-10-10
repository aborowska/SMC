function result_OPT = PF_OPT(y_true, f_loglik, f_logtrans, f_obs, f_init, f_sigma_opt, cont)
    
    M = cont.M;
    threshold = cont.threshold;
    resampl_on = cont.resampl_on;

    T = size(y_true,2);

    f_lognorm = @(xx,mm,ss) -0.5*(log(2*pi) + log(ss) + ((xx-mm).^2)./ss); % normal log density, ss is variance 
    
    % Initialization
    w = zeros(M,T+1); 
    w(:,1) = ones(M,1)/M;

    x = zeros(M,T+1);
    x(:,1) = f_init(randn(M,1));
    x_old = x;

    x_est = zeros(1,T+1);
    x_est(1,1) = sum(w(:,1).*x(:,1));
    S_est = sum(w.*(x.^2)) - x_est.^2;

    y_est = zeros(1,T);

    ESS = zeros(1,T+1);
    ESS(1,1) = 1/sum(w(:,1).^2,1);

    for ii = 2:T+1
        % GET THE NORMAL APPROXIMATION
        % mu = argmax[p(x|x_t-1)*p(y_t|x)]
        % Sigma = -1/l''(mu_opt)
%         if (mod(ii,100) == 0)
%             fprintf('Optimal candidate construction, t = %i\n',ii);
%         end        
        mu_opt = zeros(M,1);
        tic
        for jj = 1:M
            f_curr = @(xx) - f_loglik(y_true(1,ii-1),xx) - f_logtrans(xx,x(jj,ii-1));
            mu_opt(jj,1) = fminsearch(f_curr,0);
        end
        time = toc;
        fprintf('Iteration %i, opt. time %4.2f.\n',ii,time)
        % f_sigma_opt = @(xx,yy) (sigma^2).*exp(xx)./(exp(xx) + 0.5*(sigma^2).*(yy^2));
        Sigma2_opt = f_sigma_opt(mu_opt,y_true(1,ii));
        
        % SAMPLE PARTICLES
        % simulate from q=N(mu,Sigma) i.e. normal approximation
        x(:,ii) = mu_opt + sqrt(Sigma2_opt).*randn(M,1);
        x_old(:,ii) = x(:,ii); % copy in case there will be resampling

        % WEIGHTS
        % old +  loglikelihood
        % f_loglik = @(xx,vv) -0.5*(log(2*pi) + vv + (xx.^2)./(exp(vv))); % vv is the logvolatility
        w(:,ii) = log(w(:,ii-1)) + f_loglik(y_true(1,ii-1),x(:,ii)); 
        % + transition logprobability
        % f_logtrans = @(x2, x1) -0.5*(log(2*pi) + log(sigma^2) + ((x2-f_trans(x1,0)).^2)./(sigma^2));
        % f_trans = @(xx,ee) mu + phi*(xx - mu) + sigma*ee;
        w(:,ii) = w(:,ii) +  f_logtrans(x(:,ii),x(:,ii-1));
        % - log importance, i.e. log normal
        % f_lognorm = @(xx,mm,ss) -0.5*(log(2*pi) + log(ss) + ((xx-mm).^2)./ss);
        w(:,ii) = w(:,ii) - f_lognorm(x(:,ii), mu_opt, Sigma2_opt);
        max_w = max(w(:,ii));
        w(:,ii) = exp(w(:,ii) - max_w);
        w(:,ii) = w(:,ii)/sum(w(:,ii));
        ESS(1,ii) = 1/sum(w(:,ii).^2,1);

        % FILTER (before resampling to minimize the variance)
        % estimate the state 
        x_est(1,ii) = sum(w(:,ii).*x(:,ii));
        S_est(1,ii) = sum(w(:,ii).*(x(:,ii).^2)) - x_est(:,ii)^2;
        y_est(1,ii-1) = sum(w(:,ii).*f_obs(x(:,ii),1));

        % RESAMPLE
        if resampl_on
            if (ESS(1,ii) < threshold*M)
                ind_res = randsample((1:M)',M,true,w(:,ii));  
                x(:,ii) = x(ind_res,ii);
                w(:,ii) = ones(M,1)/M;;
                % the random vector x_{1:t} takes the simulated values x^{i}_{1:t} with
                % probabilities w^{i}_{1:t}
            end
        end
    end
    
    result_OPT.x_est = x_est;
    result_OPT.S_est = S_est;
    result_OPT.y_est = y_est;
    result_OPT.w = w;
    result_OPT.x = x;
    result_OPT.ESS = ESS;
end
