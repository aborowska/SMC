function result_APF = PF_APF(y_true, f_loglik, f_trans, f_obs, f_init, cont)

    M = cont.M;
    T = size(y_true,2);

    % Initialization
    w = zeros(M,T+1); 
    w(:,1) = ones(M,1)/M;

    x = zeros(M,T+1);
    x(:,1) = f_init(randn(M,1));

    x_est = zeros(1,T+1);
    x_est(1,1) = sum(w(:,1).*x(:,1));
    S_est = sum(w.*(x.^2)) - x_est.^2;
    y_est = zeros(1,T);

    ESS = zeros(1,T+1);
    ESS(1,1) = 1/sum(w(:,1).^2,1);

    for ii = 2:T+1
        % PREDICT
        % f_trans = @(xx,ee) mu + phi*(xx - mu) + sigma*ee;
        mu = f_trans(x(:,ii-1),0);

        % FIRST STAGE WEIGHTS
        % a.k.a. intermediate weights
        ind_nonzero = (w(:,ii-1) ~= 0); % to avoid taking log of zero
        w_mu = zeros(M,1);
        w_mu(ind_nonzero,:) = log(w(ind_nonzero,ii-1)) + f_loglik(y_true(1,ii-1),mu(ind_nonzero,1));
        w_mu = exp(w_mu - max(w_mu));
        w_mu = w_mu/(sum(w_mu));

        % RESAMPLE
        % draw new particles from the old particles using adapted (predicitve) weights
        ind_new = randsample((1:M)',M,true,w_mu); % sample indeces
        x_new = x(ind_new,ii-1); % resampled weights
    %     x = x(ind_new,:);
    %     w = w(ind_new,:);
        mu = mu(ind_new,1); % corresponding predictions
    %     w_mu = w_mu(ind_new,1);

        % SAMPLE 
        % draw from the transition density given the draws   
        % trans = @(xx,ee) mu + phi*(xx - mu) + sigma*ee;
        x(:,ii) = f_trans(x_new,randn(M,1));

        % SECOND STAGE WEIGHTS
        % likelihood difference between the realisation and the prediction
        w(:,ii) = f_loglik(y_true(1,ii-1),x(:,ii)) - f_loglik(y_true(1,ii-1),mu); 
        w(:,ii) = exp(w(:,ii) - max(w(:,ii)));
        w(:,ii) = w(:,ii)/sum(w(:,ii));
        ESS(1,ii) = 1./sum(w(:,ii).^2,1);

        % FILTER (before resampling to minimize the variance)
        % estimate the state 
        x_est(1,ii) = sum(w(:,ii).*x(:,ii));
        S_est(1,ii) = sum(w(:,ii).*(x(:,ii).^2)) - x_est(:,ii)^2;
        y_est(1,ii-1) = sum(w(:,ii).*f_obs(x(:,ii),1));  
    end
 
    result_APF.x_est = x_est;
    result_APF.S_est = S_est;
    result_APF.y_est = y_est;
    result_APF.w = w;
    result_APF.x = x;
    result_APF.ESS = ESS;
end
