import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike, Tuple
from typing import Callable, Any


# Implements Forward algorithm with alignment envelope for a 5-state Pair HMM (S,M,I,D,E)
# Wraps a jax.lax.scan around a jax.lax.associative_scan
# Params:
#  t(i,j) is transition from i to j where state indices are 0=S, 1=M, 2=I, 3=D, 4=E
#  e(x,y) = P(x,y)/P(x)P(y). Assumes insert and delete emission odds-ratios are 1.
def pairhmm_forward (params: Tuple,  # (t, submat)
                     xobs: ArrayLike,  # (Lx,)
                     yobs: ArrayLike,  # (Ly,)
                     env: Tuple[ArrayLike,ArrayLike,int,int,int,int],   # env = (xbegin, xend, ybegin, yend, startState, endState). xbegin, xend have shape (Ly+1)
                     ) -> ArrayLike:  # (W,Ly+1,3) where W = max(xend+2-xbegin) rounded up to nearest power of 2
    t, e = params

    assert t.shape == (5,5)
    S,M,I,D,E = range(5)

    assert e.ndim == 2
    A = e.shape[0]  # alphabet size
    assert e.shape[1] == A

    assert xobs.ndim == 1
    assert yobs.ndim == 1
    Lx = xobs.size
    Ly = yobs.size
    assert Ly > 0

    assert jnp.all(xobs < A)
    assert jnp.all(xobs >= 0)
    assert jnp.all(yobs < A)
    assert jnp.all(yobs >= 0)

    xbegin, xend, ybegin, yend, startState, endState = env
    assert xbegin.shape == (Ly+1,)
    assert xend.shape == (Ly+1,)
    assert ybegin >= 0 & ybegin < Ly & yend > ybegin & yend <= Ly
    assert startState in (S,M,I,D)
    assert endState in (M,I,D,E)
    assert jnp.all(xbegin >= 0)
    assert jnp.all(xbegin <= Lx)
    assert jnp.all(xend > xbegin)
    assert jnp.all(xend <= Lx+1)
    assert jnp.all(xbegin[1:] <= xend[:-1])
    assert jnp.all(xbegin[1:] >= xbegin[:-1])
    assert xbegin[0] == 0
    assert xend[-1] == Lx+1

    ydim = yend - ybegin

    # pad max envelope width with 2, round up to nearest power of 2
    W = jnp.max (xend - xbegin) + 2
    W = 1 << (W-1).bit_length()

    # Take transition & emission slices
    ex = jnp.pad(jnp.take(e, xobs, axis=0), (1,W-1))  # (W+Lx,A)
    t2m = t[(M,I,D),M]
    t2i = t[(M,I,D),I]
    tm2d = t[M,D]
    ti2d = t[I,D]
    td2d = t[D,D]

    # We will calculate F(i,j,k) = P(x[0:i],y[0:j],k) for xbegin[j]<=i<xend[j], ybegin<=j<=yend, 0<=k<3
    # This will be stored as F(i,j,k) = Fsparse[1+i-xbegin[j],j-ybegin,k]
    K = 3  # Fsparse will have shape (W,ydim+1,K)

    # a function created by make_scan_fn has the same type signature as scan_fn
    def make_scan_fn (j_is_one: bool) -> Callable[[Tuple[ArrayLike, int], Tuple[int, int]], Tuple[Tuple[ArrayLike, int], ArrayLike]]:
        def scan_fn (carry: Tuple[ArrayLike, int],  # (Fprev, xbegin_prev)
                     jInfo: Tuple[int, int]) -> Tuple[Tuple[ArrayLike, int], ArrayLike]:  # (F, xbegin_j), F
            Fprev, xbegin_prev = carry
            yTok, xbegin_j = jInfo
            # Line up Fprev_ins so its incoming insert states are aligned with ours,
            # and Fprev_mat so its incoming match states are aligned with ours
            xshift = xbegin_j - xbegin_prev
            Fprev_ins = jax.lax.dynamic_slice_in_dim(Fprev, xshift, W - xshift, axis=0)
            Fprev_ins = jnp.pad(Fprev_ins, ((0, xshift), (0, 0)))  # (W,K)
            Fprev_mat = jnp.roll(Fprev_ins, 1, axis=0)  # (W,K)
            # Similarly, line up ex
            ex_j = jax.lax.dynamic_slice_in_dim(ex[:,yTok], xbegin_j, W, axis=0)  # (W,)
            # Calculate incoming match and insert
            Fm = jnp.einsum ('i,ik,k->i', ex_j, Fprev_mat, t2m)  # (W,)
            Fi = jnp.einsum ('ik,k->i', Fprev_ins, t2i)  # (W,)
            # Here we want the jax in-place array modification equivalent of
            # if xbegin_j == 0 and j == 1 then Fi[0] = t[S,I]
            # if xbegin_j <= 1 and j == 1 then Fm[1-xbegin_j] = t[S,M] * ex[0,yTok]
            if j_is_one:
                Fi = jax.lax.cond(
                        xbegin_j == 0,
                        lambda Fi: Fi.at[0].set(t[startState, I]),
                        lambda Fi: Fi,
                        Fi
                    )
                Fm = jax.lax.cond(
                        xbegin_j <= 1,
                        lambda Fm: Fm.at[1 - xbegin_j].set(t[startState, M] * ex[0, yTok]),
                        lambda Fm: Fm,
                        Fm
                    )        
            # The recurrence for the Fd states is
            #  Fd[i] = Fm[i-1] * tm2d + Fi[i-1] * ti2d + Fd[i-1] * td2d  for i>0, Fd[0] = 0
            # We will calculate this as a matrix product:
            # Fd_vec[i] = (Fd[i] 1), with Fd_vec[0] = (0 1), and Fd_vec[i+1] = Fd_vec[i] Fd_mx[i] with
            # Fd_mx[i] = (td2d     0)
            #            (Fmi2d[i] 1)
            #  where Fmi2d[i] = Fm[i] * tm2d + Fi[i] * ti2d
            # Hence Fd[i] = (prod_{j=0}^{i-1} Fd_mx[j])[1,0]  for i>0
            # We can represent any partial matrix product by the first column (a,b) (the second column of the product is always (0,1))
            #  then (a1,b1) * (a2,b2) = (a1 0) (a2 0) = (a1*a2    0) = (a1*a2,b1*a2+b2)
            #                           (b1 1) (b2 1)   (b1*a2+b2 1)
            # Component b then contains our final product
            #   Fd[i] = ( prod_{j=0}^{i-1} (td2d,Fmi2d[i]) )[1]
            def Fd_mx_mul (f1: ArrayLike,  # (2,)
                           f2: ArrayLike):  # (2,)
                x1, y1 = f1[0], f1[1]
                x2, y2 = f2[0], f2[1]
                return jnp.array([x1*x2, y1*x2+y2])  # (2,)

            Fmi2d = Fm * tm2d + Fi * ti2d  # (W,)
            Fd_mx = jnp.stack([td2d * jnp.ones_like(Fmi2d), Fmi2d], axis=-1)  # (W,2)
            Fd_mx_prod = jax.lax.associative_scan (Fd_mx_mul, Fd_mx)  # (W,2)
            Fd = jnp.pad(Fd_mx_prod[:-1,1],(1,0)) # (W,)  # Fd[0] = 0, Fd[1] = Fmi2d[0], Fd[1:] = Fd_mx_prod[:-1,1]
            # Combine match, insert, and delete
            F = jnp.stack([Fm, Fi, Fd], axis=-1)  # (W,3)
            # Return carry for next iteration, and F as output
            return (F, xbegin_j), F
        return scan_fn
    
    # First row of F (j=0) just has transitions from start->delete->delete....
    Fd0 = jnp.pad(jnp.pow (t[D,D], jnp.arange(xend[0])) * t[startState,D], (1,W-xend[0]-1))  # (W,)
    F0 = jnp.stack([jnp.zeros_like(Fd0), jnp.zeros_like(Fd0), Fd0], axis=-1)  # (W,3)

    # Second row (j=1) is special, contains transitions from S state on first row
    make_row1 = make_scan_fn (j_is_one=True)
    _, F1 = jax.lax.cond (
                ydim > 0,
                lambda: make_row1((F0, xbegin[ybegin]), (yobs[ybegin], xbegin[ybegin+1])),
                lambda: (0, jnp.array([]))  # empty array if ydim < 1
            )    


    # Now scan over yobs
    scan_row = make_scan_fn (j_is_one=False)
    _, F = jax.lax.cond (
                ydim > 1,
                lambda: jax.lax.scan (scan_row, (F1, xbegin[ybegin+1], 0), (yobs[ybegin+1:], xbegin[ybegin+2:]), length=ydim-1),
                lambda: (0, jnp.array([]))  # empty array if ydim < 2
            )    

    Fsparse = jnp.stack([F0, F1, F], axis=0)  # (ydim+1,W,3)
    Fend = jnp.dot(Fsparse[yend-ybegin,Lx-xbegin[yend],:], t[(M,I,D),endState])

    return Fsparse, Fend


def getf (Fsparse: ArrayLike,  # (Ly+1,W,3)
          Fend: ArrayLike,  # (1,)
          env: Tuple[ArrayLike,ArrayLike,int,int,int,int],   # env = (xbegin, xend, ybegin, yend, startState, endState). xbegin, xend have shape (Ly+1)
          t: ArrayLike,  # (5,5)
          i, j, k) -> ArrayLike:
    S,M,I,D,E = range(5)

    assert Fsparse.ndim == 3
    assert Fsparse.shape[2] == 3
    Ly = Fsparse.shape[0] - 1
    W = Fsparse.shape[1]

    assert Fend.shape == (1,)

    xbegin, xend, ybegin, yend, startState, endState = env
    assert xbegin.shape == (Ly+1,)
    assert xend.shape == (Ly+1,)
    Lx = xend[-1] - 1
    assert ybegin >= 0 & ybegin < Ly & yend > ybegin & yend <= Ly
    assert startState in (S,M,I,D)
    assert endState in (M,I,D,E)

    assert t.shape == (5,5)

    # TODO: accept a precomputed set of cell coordinates for the constrained part of the state path (<ybegin or >yend),
    # along with the path probabilities to, or from, those cells, and use them to compute the requested cell.
    return jax.lax.cond ((i == 0) & (j == 0) & (k == 0),   # start state
                         lambda: 1.0,
                         lambda: jax.lax.cond(
                            (i == 0) & (j == 0) & (Lx == 0) & (Ly == 0) & (k == 4),  # end state, empty sequence
                            lambda: t(S,E),
                            lambda: jax.lax.cond(
                                (i == Lx) & (j == Ly) & (k == 4),  # end state, nonempty sequence
                                lambda: jnp.dot(Fsparse[Ly,Lx-env[0,Ly],:], t[(M,I,D),E])),
                            lambda: jax.lax.cond(
                                (j >= 0) & (j < Fsparse.shape[0]) & (i >= env[0,j]) & (i < env[1,j]) & (k > 0) & (k < 4),  # inside envelope
                                lambda: Fsparse[j,i-env[0,j],k],
                                lambda: 0.0)))
