// Convert from GGI params (insertion & deletion rates (insrate,delrate) & extension probs (insextprob,delextprob))
// to TKF92 params (fragment insertion & deletion rates (lam,mu) and fragment extension prob (r)).
// We make use of the following identities:
//  delrate / (1 - delextprob)  = mu                      = 1 / reslife       where  reslife = mean time to deletion of any given residue
//  delextprob/insextprob - 1   = (mu/lam - 1) * (1 - r)  = 1 / seqlen        where   seqlen = mean sequence length at equilibrium
//  insextprob                  = r                       = 1 - 1 / inslen    where   inslen = mean insertion event length
const ggiParamsToTkf92Params = (ggiParams) => {
    const { delrate, insextprob, delextprob } = ggiParams;
    const r = insextprob;
    const mu = delrate / (1 - delextprob);
    const lam = mu * (1 - r) / (delextprob / insextprob - r);
    return { lam, mu, r };
};

// Convert from expectations (mean eqm sequence length, mean insertion event length, mean lifespan of residue) to GGI params (insertion & deletion rates & extension probs)
// We are making use of the following identities:
//  1/inslen = 1 - insextprob
//  1/seqlen = delextprob/insextprob + 1
//  1/reslife = delrate / (1 - delextprob)
//  insrate * delextprob * (1 - insextprob) = delrate * insextprob * (1 - delextprob)    [reversibility/equilibrium condition]
export const expectationsToGgiParams = (seqlen, inslen, reslife) => {
    const insextprob = 1 - 1 / inslen;  // GGI parameter x
    const delextprob = insextprob * (seqlen + 1) / seqlen;  // GGI parameter y
    const delrate = (1 / reslife) * (1 - delextprob);  // GGI parameter mu_0
    const insrate = delrate * insextprob * (1 - delextprob) / (delextprob * (1 - insextprob));   // GGI parameter lambda_0
    return { insrate, delrate, insextprob, delextprob };
};


// Compute various TKF92 equilibrium probabilities based on GGI parameters
export const calcTkf92EqmProbs = (ggiParams) => {
    const { lam, mu, r } = ggiParamsToTkf92Params(ggiParams);
    const kappa = lam / mu;
    const nu = r + (1 - r) * kappa;
    return { lam, mu, r, kappa, nu };
};

// Compute various TKF92 time-dependent probabilities based on GGI parameters
export const calcTkf92TransProbs = (t, ggiParams) => {
    const { lam, mu, r, kappa, nu } = calcTkf92EqmProbs(ggiParams);
    // Calculate alpha, beta, gamma for TKF92 model
    const exp_mu_t = Math.exp(-mu * t);
    const exp_lam_t = Math.exp(-lam * t);

    const alpha = exp_mu_t;
    const beta = (lam * (exp_lam_t - exp_mu_t)) / (mu * exp_lam_t - lam * exp_mu_t);
    const gamma = 1 - mu * beta / (lam * (1 - alpha));

    return { lam, mu, r, kappa, nu, alpha, beta, gamma };
};

// Transition matrix for Pair HMM on pseudo-branch "above" root
export const tkf92RootTransitionMatrix = (ggiParams) => {
    const { kappa, nu } = calcTkf92EqmProbs(ggiParams);
    return [[0, 0, kappa, 0, 1-kappa],
            [0, 0, 0, 0, 0],
            [0, 0, nu, 0, 1-nu],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
};

// Transition matrix for Pair HMM on branch of length t
export const tkf92BranchTransitionMatrix = (t, ggiParams) => {
    const { lam, mu, r, kappa, nu, alpha, beta, gamma } = calcTkf92TransProbs(t, ggiParams);
    // States are S, M, I, D, E
    return [[0, (1-beta)*alpha, beta, (1-beta)*(1-alpha), 1-beta],
            [0, (r + (1-r)*(1-beta)*kappa*alpha)/nu, (1-r)*beta, (1-r)*(1-beta)*kappa*(1-alpha)/nu, 1-beta],
            [0, (1-r)*(1-beta)*kappa*alpha/nu, r + (1-r)*beta, (1-r)*(1-beta)*kappa*(1-alpha)/nu, 1-beta],
            [0, (1-r)*(1-gamma)*kappa*alpha/nu, (1-r)*gamma, (r + (1-r)*(1-gamma)*kappa*(1-alpha))/nu, 1-gamma],
            [0, 0, 0, 0, 0]];
};
