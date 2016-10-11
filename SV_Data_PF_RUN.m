clear all
close all

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s); 
addpath(genpath('include/'));
addpath(genpath('PF/'));

model = 'sv';
data_on = false;

% mu_init = [0.5, 0.98, 0.15^2];
load('SML_ibm.mat', 'par_SV_opt', 'V_SV_corr_opt');

mu = par_SV_opt(1,1);
phi = par_SV_opt(1,2);
sigma = sqrt(par_SV_opt(1,3));
sigma_init = sigma/sqrt(1-phi^2);

SS = 20;
MSE_BF_mat = zeros(1,SS);
MSE_SISR_mat = zeros(1,SS);
MSE_APF_mat = zeros(1,SS);
MSE_NAIS_mat = zeros(1,SS);

for ss = 1:SS
    if data_on
        y = csvread('IBM_ret.csv');
        y = 100*y';
        T = size(y,2);
    else
        % if not data then simulate the true processes
        f_trans = @(xx,ee) mu + phi*(xx - mu) + sigma*ee;
        f_obs = @(xx,ee) exp(xx/2).*ee;

        T = 1500;

        x_true = zeros(1,T+1);
        eta_true = randn(1,T);
        eps_true = randn(1,T);

        x_true(1,1) = sigma_init*randn(1,1);
        for ii = 2:T+1
            x_true(1,ii) = f_trans(x_true(1,ii-1), eta_true(1,ii-1));
        end
        y = f_obs(x_true(2:T+1), eps_true);
    end

    %% PFs 
    f_loglik = @(xx,vv) -0.5*(log(2*pi) + vv + (xx.^2)./(exp(vv))); % vv is the logvolatility
    f_trans = @(xx,ee) mu + phi*(xx - mu) + sigma*ee;
    f_obs = @(xx,ee) exp(xx/2).*ee;
    f_init = @(xx) sigma_init.*xx;

    cont_PF.M = 1000;
    cont_PF.threshold =  1;
    cont_PF.resampl_on = 1;

    result_BF = PF_SISR(y, f_loglik, f_trans, f_obs, f_init, cont_PF);
    result_APF = PF_APF(y, f_loglik, f_trans, f_obs, f_init, cont_PF);

    cont_PF.threshold =  0.5;
    result_SISR = PF_SISR(y, f_loglik, f_trans, f_obs, f_init, cont_PF);

    % f_logtrans = @(x2, x1) -0.5*(log(2*pi) + log(sigma^2) + ((x2-f_trans(x1,0)).^2)./(sigma^2));
    % f_sigma_opt = @(xx,yy) (sigma^2).*exp(xx)./(exp(xx) + 0.5*(sigma^2).*(yy^2));
    % cont_PF.threshold =  0.5;
    % result_OPT = PF_OPT(y, f_loglik, f_logtrans, f_obs, f_init, f_sigma_opt, cont_PF);

    %% NAIS 
    y = y';
    T = size(y,1);
    par_NAIS_init.b = zeros(T,1);
    par_NAIS_init.C = ones(T,1); 

    cont = EMitISEM_Control(model);
    cont_NAIS = cont.nais;
    clear cont
    % kernel = @(a) posterior_sv(y, a, par_NAIS_init, prior_const, cont.nais);
    [~, result_NAIS.x_est] = NAIS_param(par_NAIS_init, y, par_SV_opt, cont_NAIS); % Efficient importance parameters via NAIS

    %% MSE
    MSE_BF = mean((x_true(1,2:T+1) - result_BF.x_est(1,2:T+1)).^2);
    MSE_SISR = mean((x_true(1,2:T+1) - result_SISR.x_est(1,2:T+1)).^2);
    MSE_APF = mean((x_true(1,2:T+1) - result_APF.x_est(1,2:T+1)).^2);
    MSE_NAIS = mean((x_true(1,2:T+1) - result_NAIS.x_est').^2);

    text_BF = ['MSE\_BF = ',sprintf('%6.4f',MSE_BF),','];
    text_SISR = ['MSE\_SISR = ',sprintf('%6.4f',MSE_SISR),','];
    text_APF = ['MSE\_APF = ',sprintf('%6.4f',MSE_APF),','];
    text_NAIS = ['MSE\_NAIS = ',sprintf('%6.4f',MSE_NAIS),','];
    str = {text_BF, text_SISR, text_APF, text_NAIS};

    MSE_BF_mat(1,ss) = MSE_BF;
    MSE_SISR_mat(1,ss) = MSE_SISR;
    MSE_APF_mat(1,ss) = MSE_APF;
    MSE_NAIS_mat(1,ss) = MSE_NAIS;
end
%% Figures
figure(1)
set(gcf,'units','normalized','outerposition',[0 0 1 1]);   

hold on
plot(1:T,x_true(1,2:T+1),'k')
plot(1:T,result_BF.x_est(1,2:T+1),'b')
plot(1:T,result_SISR.x_est(1,2:T+1),'g')
plot(1:T,result_APF.x_est(1,2:T+1),'r')
plot(1:T,result_NAIS.x_est,'m')
hold off
legend('true','BF','SISR 0.5','APF','NAIS')
annotation('textbox', [0.15,0.8,0.1,0.1],'String', str);


figure(2)
hold on
plot(1:T,y,'k')
plot(1:T,result_BF.y_est,'b')
plot(1:T,result_SISR.y_est,'g')
plot(1:T,result_APF.y_est,'r')
hold off
legend('obs','BF','SISR 0.5','APF')


figure(3)
hold on
plot(1:T+1,result_BF.ESS/cont_PF.M,'b')
plot(1:T+1,result_SISR.ESS./cont_PF.M,'g')
plot(1:T+1,result_APF.ESS/cont_PF.M,'r')
plot(1:T+1,cont_PF.threshold*ones(1,T+1),'k')
hold off
legend('BF','SISR 0.5','APF')


figure(4)
bin = 4;
bm = 10; % bin mesh
bin_grid = (-bin : 1/bm : bin)';
epdf = zeros(2*bm*bin+1,T+1);
for ii = 2:T+1
    for kk = 1:2*bm*bin
        for jj = 1:M
            if (bin_grid(kk,1) <= x_old(jj,ii)) && (x_old(jj,ii) < bin_grid(kk+1,1))
                epdf(kk,ii) = epdf(kk,ii) + 1;
            end
        end
    end
end
E = epdf/M;
E_plot = E(:,2:50:T+1)';
waterfall(E_plot)
ribbon(E_plot)
