export const logSumExp = (arr) => {
  const max = Math.max(...arr);
  return max + Math.log(arr.reduce((sum, val) => sum + Math.exp(val - max), 0));
}

export const logHypergeom2F1 = (a, b, c, z) => {
  if (!(a <= 0 && Math.floor(a) === a && b <= 0 && Math.floor(b) === b))
    throw new Error('a and b must be non-positive integers');
  if (!(c >= 0 && Math.floor(c) === c))
    throw new Error('c must be a non-negative integer');
  
  const max_k = Math.min(-a, -b);
  const log_z = Math.log(z);
  
  let poch_a = 0;  // log (a)_k
  let poch_b = 0;  // log (b)_k
  let poch_c = 0;  // log (c)_k
  let k_factorial = 0; // log(k!) = log (1)_k
  let pow = 0;     // log z^k
  let sum = [pow + poch_a + poch_b - poch_c - k_factorial]; // Initial term for k=0
  
  for (let k = 1; k <= max_k; k++) {
    poch_a += Math.log(1 - a - k);
    poch_b += Math.log(1 - k - b);
    poch_c += Math.log(c + k - 1);
    k_factorial += Math.log(k);
    pow += log_z;
    sum.push (pow + poch_a + poch_b - poch_c - k_factorial); // Add term for current k
  }

  return logSumExp(sum);
}
