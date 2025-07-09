// Approximate GGI model with TKF92 model
const ggiParamsToTkfParams = (indelParams) => {
    const [ggi_lam,ggi_mu,ggi_x,ggi_y] = indelParams;
    const r = ggi_x;
    const mu = ggi_mu / (1 - ggi_y);
    const lam = (1 - r) / (ggi_y / ggi_x - r);
    return { lam, mu, r };
};

export const tkf92RootTransitionMatrix = (indelParams) => {
    const { lam, mu, r } = ggiParamsToTkfParams(indelParams);
    const kappa = lam / mu;
    const nu = r + (1 - r) * kappa;
    return [[0, 0, kappa, 0, 1-kappa],
            [0, 0, 0, 0, 0],
            [0, 0, nu, 0, 1-nu],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
};
  
export const tkf92BranchTransitionMatrix = (t, indelParams) => {
    const { lam, mu, r } = ggiParamsToTkfParams(indelParams);
    // Calculate alpha, beta, gamma for TKF92 model
    const exp_mu_t = Math.exp(-mu * t);
    const exp_lam_t = Math.exp(-lam * t);

    const alpha = exp_mu_t;
    const beta = (lam * (exp_lam_t - exp_mu_t)) / (mu * exp_lam_t - lam * exp_mu_t);
    const gamma = 1 - mu * beta / (lam * (1 - alpha));

    const kappa = lam / mu;
    const nu = r + (1 - r) * kappa;

    // States are S, M, I, D, E
    return [[0, (1-beta)*alpha, beta, (1-beta)*(1-alpha), 1-beta],
            [0, (r + (1-r)*(1-beta)*kappa*alpha)/nu, (1-r)*beta, (1-r)*(1-beta)*kappa*(1-alpha)/nu, 1-beta],
            [0, (1-r)*(1-beta)*kappa*alpha/nu, r + (1-r)*beta, (1-r)*(1-beta)*kappa*(1-alpha)/nu, 1-beta],
            [0, (1-r)*(1-gamma)*kappa*alpha/nu, (1-r)*gamma, (r + (1-r)*(1-gamma)*kappa*(1-alpha))/nu, 1-gamma],
            [0, 0, 0, 0, 0]];
};
