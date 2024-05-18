// calculate L, M
const lm = (t, rate, prob) => Math.exp (-rate * t / (1 - prob));
const indels = (t, rate, prob) => 1 / lm(t,rate,prob) - 1;

// calculate derivatives of (a,b,u,q)
const derivs = (t, counts, params) => {
  const [lam,mu,x,y] = params;
  const [a,b,u,q] = counts;
  const L = lm (t, lam, x);
  const M = lm (t, mu, y);
  const num = mu * (b*M + q*(1.-M));
  const unsafe_denom = M*(1.-y) + L*q*y + L*M*(y*(1.+b-q)-1.);
  const denom = unsafe_denom > 0 ? unsafe_denom : 1;
  const one_minus_m = M < 1 ? 1 - M : Number.MIN_VALUE;
  return unsafe_denom > 0 ? [mu*b*u*L*M*(1.-y)/denom - (lam+mu)*a,
                            -b*num*L/denom + lam*(1.-b),
                            -u*num*L/denom + lam*a,
                            ((M*(1.-L)-q*L*(1.-M))*num/denom - q*lam/(1.-y))/one_minus_m]
                    : [-lam-mu,lam,lam,0];
}

// test whether time is past threshold of alignment signal being undetectable
const alignmentIsProbablyUndetectable = (t, indelParams, alphabetSize) => {
    const [lam,mu,x,y] = indelParams;
    const expectedMatchRunLength = 1. / (1. - Math.exp(-mu*t));
    const expectedInsertions = indels(t,lam,x);
    const expectedDeletions = indels(t,mu,y);
    const kappa = 2.;
    return t > 0. && ((expectedInsertions + 1) * (expectedDeletions + 1)) > kappa * Math.pow(alphabetSize, expectedMatchRunLength);
};
    
// initial transition matrix
const zeroTimeTransitionMatrix = (indelParams) => {
  const [lam,mu,x,y] = indelParams;
  return [[1.,0.,0.],
          [1.-x,x,0.],
          [1.-y,0.,y]];
};

// convert counts (a,b,u,q) to transition matrix ((a,b,c),(f,g,h),(p,q,r))
const smallTimeTransitionMatrix = (t, indelParams) => {
    const [lam,mu,x,y] = indelParams;
    const steps = 100;
    const dt0 = 1e-6;
    const [a,b,u,q] = integrateCounts(t,indelParams,steps,dt0);
    const L = lm(t,lam,x);
    const M = lm(t,mu,y);
    return [[a,b,1-a-b],
            [u*L/(1-L),1-(b+q*(1-M)/M)*L/(1-L),(b+q*(1-M)/M-u)*L/(1-L)],
            [(1-a-u)*M/(1-M),q,1-q-(1-a-u)*M/(1-M)]];
};

// get limiting transition matrix for large times
const largeTimeTransitionMatrix = (t, indelParams) => {
    const [lam,mu,x,y] = indelParams;
    const g = 1. - lm(t,lam,x);
    const r = 1. - lm(t,mu,y);
    return [[(1-g)*(1-r),g,(1-g)*r],
            [(1-g)*(1-r),g,(1-g)*r],
            [(1-r),0,r]];
};

// get transition matrix for any given time
export const transitionMatrix = (t, indelParams, alphabetSize) => {
    const [lam,mu,x,y] = indelParams;
    return t > 0. ? (alignmentIsProbablyUndetectable(t,indelParams,alphabetSize || 20)
                                 ? largeTimeTransitionMatrix(t,indelParams)
                                 : smallTimeTransitionMatrix(t,indelParams))
                    : zeroTimeTransitionMatrix(indelParams);
};

export const dummyRootTransitionMatrix = () => {
  return [[0,1,0],[1,1,0],[1,0,0]];
};

// Runge-Kutta integration
const integrateCounts = (t, params, steps, dt0) => {
  const [lam,mu,x,y] = params;
  const T0 = [1.,0.,0.,0.];
  const dt0_abs = Math.min (t/steps, dt0 / Math.min(1.,1./Math.max(lam,mu)));
  const dlog = Math.log (t/dt0_abs) / steps;
  const ts = Array.from({length: steps}, (_,i) => Math.exp(dlog*i) * dt0_abs);
  const ts_with_0 = [0].concat (ts);
  const dts = ts.map ((t,i) => t - ts_with_0[i]);
  const RK4body = (T, dt, n) => {
    const t = ts_with_0[n];
    const compute_derivs = (k, f) => {
        let Tnew = T.slice(0);
        for (let i = 0; i < T.length; i++)
            Tnew[i] += dt * k[i] * f;
        return derivs(t+dt*f, Tnew, params);
    };
    const k1 = derivs (t, T, params);
    const k2 = compute_derivs (k1, 1/2);
    const k3 = compute_derivs (k2, 1/2);
    const k4 = compute_derivs (k3, 1);
    for (let i = 0; i < T.length; i++)
        T[i] += dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6;
    return T;
  };
  return dts.reduce (RK4body, T0);
};
