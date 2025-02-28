export class PairwiseAlignment {
  private pairs: [number, number][];
  private xToY: (number | null)[];
  private yToX: (number | null)[];

  constructor(alignment: [number, number][], private Lx: number, private Ly: number) {
    this.pairs = [...alignment].sort((a, b) => a[0] - b[0]);
    this.xToY = Array(Lx + 1).fill(null);
    this.yToX = Array(Ly + 1).fill(null);
    
    for (const [x, y] of this.pairs) {
      if (this.xToY[x] !== null || this.yToX[y] !== null) {
        throw new Error('Invalid alignment: duplicate x or y value');
      }
      this.xToY[x] = y;
      this.yToX[y] = x;
    }
  }

  size(): number {
    return this.pairs.length;
  }

  get_nth(n: number): [number, number] {
    if (n < 1 || n > this.pairs.length) {
      throw new Error('Index out of bounds');
    }
    return this.pairs[n - 1];
  }

  get_y_for_x(x: number): number | null {
    return this.xToY[x] ?? null;
  }

  get_x_for_y(y: number): number | null {
    return this.yToX[y] ?? null;
  }

  insert(pair: [number, number]): void {
    const [x, y] = pair;
    if (this.xToY[x] !== null || this.yToX[y] !== null) {
      throw new Error('Insertion violates alignment constraints');
    }
    
    let i = this.pairs.findIndex(([px]) => px > x);
    if (i === -1) i = this.pairs.length;
    
    if (
      (i > 0 && (this.pairs[i - 1][0] >= x || this.pairs[i - 1][1] >= y)) ||
      (i < this.pairs.length && (this.pairs[i][0] <= x || this.pairs[i][1] <= y))
    ) {
      throw new Error('Insertion violates ordering constraints');
    }
    
    this.pairs.splice(i, 0, pair);
    this.xToY[x] = y;
    this.yToX[y] = x;
  }

  transpose(): PairwiseAlignment {
    return new PairwiseAlignment(this.pairs.map(([x, y]) => [y, x]), this.Ly, this.Lx);
  }

  compose(other: PairwiseAlignment): PairwiseAlignment {
    const newPairs: [number, number][] = [];
    
    for (const [x, y] of this.pairs) {
      const z = other.get_y_for_x(y);
      if (z !== null) {
        newPairs.push([x, z]);
      }
    }
    
    return new PairwiseAlignment(newPairs, this.Lx, other.Ly);
  }
}
