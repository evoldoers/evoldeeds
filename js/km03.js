const transitionMatrix = (t, indelParams) => {
    if (t === 0)
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const [lam, mu, x, y] = indelParams;
    const r = (lam + mu) / 2;
    const a = (x + y) / 2;
    const Pid = 1 - Math.exp(-2*r*t);
    const Pid_prime = 1 - (1 - Math.exp(-2*r*t)) / (2*r*t);
    const T00 = 1 - Pid*(1-Pid_prime*(1-a)/(4+4*a));
    const T01 = (1-T00)/2;
    const T02 = T01;
    const E10 = 1-a + Pid_prime*a*(1-a)/(2+2*a) - Pid*(7-7*a)/8;
    const E11 = a + Pid_prime*a*a/(1-a*a) + Pid*(1-a)/2;
    const E12 = Pid_prime*a*a/(2+2*a) + Pid*(3-3*a)/8;
    const E1 = 1 + Pid_prime*a/(2-2*a);
    const T10 = E10/E1;
    const T11 = E11/E1;
    const T12 = E12/E1;
    const T20 = T10;
    const T22 = T11;
    const T21 = T12;
    return [[T00, T01, T02],
            [T10, T11, T12],
            [T20, T21, T22]];
};

module.exports = { transitionMatrix };