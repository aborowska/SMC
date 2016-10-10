function result_SISR = PF_SISR(y_true, f_loglik, f_trans, f_obs, f_init, cont)

    M = cont.M;
    threshold = cont.threshold;
    resampl_on = cont.resampl_on;

    T = size(y_true,2);

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
        % SAMPLE PARTICLES
        % trans = @(xx,ee) mu + phi*(xx - mu) + sigma*ee;
        x(:,ii) = f_trans(x(:,ii-1),randn(M,1));
        x_old(:,ii) = x(:,ii); % copy in case there will be resampling

        % WEIGHTS
        w(:,ii) = log(w(:,ii-1)) + f_loglik(y_true(1,ii-1),x(:,ii)); % bootstrap --> update is just the likelihood 
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
    
    result_SISR.x_est = x_est;
    result_SISR.S_est = S_est;
    result_SISR.y_est = y_est;
    result_SISR.w = w;
    result_SISR.x = x;
    result_SISR.ESS = ESS;
    result_SISR.x_old = x_old;
end
