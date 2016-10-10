#include "mex.h"
#include "math.h"

void KFS_mex(double *y, mwSignedIndex n, mwSignedIndex S, double *P1, double *c, double *H, double *Q,
        double *d, double *T, double *R, double *Z,
        double *theta_smooth, double *V, double *nu, double *F_inv, double *eps, double *K, double *L)
{
    double *a, *P;
    double *F;
    double *D, *u, *N, *r;
    mwSignedIndex i, j, ind;
    
    
    /* Variable size arrays */
    a = malloc((n+1)*(S)*sizeof(double));    
    P = malloc((n+1)*(S)*sizeof(double));
    F = malloc((n*S)*sizeof(double));    
    D = malloc((S)*sizeof(double));   
    u = malloc((S)*sizeof(double)); 
    N = malloc((S)*sizeof(double));    
    r = malloc((S)*sizeof(double));        
    
    /* Initialise */
    for (j=0; j<S; j++)
    {
        a[j*(n+1)] = 0;
        P[j*(n+1)] = P1[j];
/*        V[j*S+n-1] = 0;*/
        N[j] = 0;
        r[j] = 0;
    }
    
    /* Kalman filter */
    for (i=0; i<n; i++) 
    {
        for (j=0; j<S; j++)
        {   
            ind = j*n + i;
            nu[ind] = y[ind] - c[j] - Z[0]*a[ind+j];
            F[ind] = Z[0]*Z[0]*P[ind+j] + H[ind];
            F_inv[ind] = 1/F[ind];
            K[ind] = T[j]*P[ind+j]*Z[0]/F[ind];
            L[ind] = T[j] - K[ind]*Z[0];
            a[ind+j+1] = d[0] + T[j]*a[ind+j] + K[ind]*nu[ind];
            P[ind+j+1] = T[j]*T[j]*P[ind+j] + R[0]*R[0]*Q[j] - K[ind]*K[ind]*F[ind];                
        }                  
     }
    
    /* Disturbance smoothing JSDK */
    
    for (i=n-1; i>=0; i--) 
    {
        for (j=0; j<S; j++)
        { 
            ind = j*n + i;
            D[j] = F_inv[ind] + K[ind]*N[j]*K[ind];   
            V[ind] = H[ind] - H[ind]*D[j]*H[ind];
            u[j] = nu[ind]*F_inv[ind] - K[ind]*r[j];
            eps[ind] = H[ind]*u[j];
            theta_smooth[ind] = y[ind] - eps[ind];
            r[j] = Z[0]*nu[ind]*F_inv[ind] + L[ind]*r[j];
            N[j] = Z[0]*Z[0]*F_inv[ind] + L[ind]*N[j]*L[ind];            
        }
     }
    
    /* Free allocated memory */
    free(a); free(P); free(F); free(D); free(u); free(N); free(r);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    mwSignedIndex n, S;                                         /* size of matrix */
    double *y, *P1, *c, *H, *Q, *d, *T, *R, *Z;                 /* input*/
    double *theta_smooth, *V, *nu, *F_inv, *eps, *K, *L;        /* output */

    
    /* Getting the inputs */
    y = mxGetPr(prhs[0]);
    P1 = mxGetPr(prhs[1]);
    c = mxGetPr(prhs[2]);
    H = mxGetPr(prhs[3]);
    Q = mxGetPr(prhs[4]);
    d = mxGetPr(prhs[5]);
    T = mxGetPr(prhs[6]);
    R = mxGetPr(prhs[7]);
    Z = mxGetPr(prhs[8]);
    
    n = mxGetM(prhs[3]); /* no. of observations */
    S = mxGetN(prhs[3]); /* no of parameter draws */
    
    /* create the output matrices */
    plhs[0] = mxCreateDoubleMatrix(n,S,mxREAL); 
    plhs[1] = mxCreateDoubleMatrix(n,S,mxREAL); 
    plhs[2] = mxCreateDoubleMatrix(n,S,mxREAL); 
    plhs[3] = mxCreateDoubleMatrix(n,S,mxREAL); 
    plhs[4] = mxCreateDoubleMatrix(n,S,mxREAL); 
    plhs[5] = mxCreateDoubleMatrix(n,S,mxREAL); 
    plhs[6] = mxCreateDoubleMatrix(n,S,mxREAL); 
    
    /* get a pointer to the real data in the output matrix */
    theta_smooth = mxGetPr(plhs[0]);
    V = mxGetPr(plhs[1]);
    nu = mxGetPr(plhs[2]);
    F_inv = mxGetPr(plhs[3]);
    eps = mxGetPr(plhs[4]);
    K = mxGetPr(plhs[5]);    
    L = mxGetPr(plhs[6]);
    
    /* call the function */
    KFS_mex(y, n, S, P1, c, H, Q, d, T, R, Z, theta_smooth, V, nu, F_inv, eps, K, L);
  
}
