function cont =  EMitISEM_Control(model)

    %% MitISEM Control
    cont.mit.N = 10000; %1e5;
    cont.mit.Hmax = 10;
    % cont.mit.weight_err = 1e-05;
    % cont.mit.weight_frac = 0.25;
    cont.mit.CV_tol = 0.1; %0.01
    cont.mit.CV_old = 100;

    cont.mit.ISpc = 0.1;
    cont.mit.pnc = 0.1; % probability of a new component
    cont.mit.dfnc = 5; % degrees of freedom of a new component
    cont.mit.tol_pr = 0;

    cont.mit.norm = true;

    cont.EM.maxit = 1000;
    cont.EM.tol = 0.001;

    cont.df.maxit = 1000;
    cont.df.opt = true;
    cont.df.range = [1,10];
    cont.df.tol = eps^(0.25);

    cont.resmpl_on = false;
    cont.mit.iter_max = 2;

    %% NAIS control parameters
    if (exist('model','var') && (~isempty(strfind(model, 'sv'))))
        cont.nais.M = 20; % number of the Gauss-Hermite nodes
        cont.nais.tol = 0.0001; % convergence tolerance
        cont.nais.tol_C = 1e-5; % tolerance for the variance 
        [cont.nais.GH.z, cont.nais.GH.h]  = hernodes(cont.nais.M);
        cont.nais.GH.z = -cont.nais.GH.z;
        cont.nais.GH.h = cont.nais.GH.h.*exp(0.5*(cont.nais.GH.z).^2);
        cont.nais.iter_max = 20;
        if isempty(strfind(model, 't'))
            cont.nais.err = 'n';
        else
            cont.nais.err = 't';
        end
        cont.nais.data_on = 'est'; %'simt'

        cont.nais.HP = 4; % number of the last states returned by the simulation smoother (+1)
    end
end