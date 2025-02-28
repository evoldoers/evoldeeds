import * as fs from 'fs';
import * as path from 'path';
import { PairwiseAlignment } from '../pairalign';

describe('PairwiseAlignment', () => {
  let testData: any;

  beforeAll(() => {
    const testFile = path.join(__dirname, 'test_data.json');
    testData = JSON.parse(fs.readFileSync(testFile, 'utf8'));
  });

  test('Valid alignments should initialize correctly', () => {
    const { alignment, Lx, Ly } = testData.valid_case;
    const pa = new PairwiseAlignment(alignment, Lx, Ly);
    expect(pa.size()).toBe(alignment.length);
  });

  test('get_nth should return correct pair', () => {
    const { alignment, Lx, Ly } = testData.valid_case;
    const pa = new PairwiseAlignment(alignment, Lx, Ly);
    expect(pa.get_nth(1)).toEqual(alignment[0]);
  });

  test('get_y_for_x and get_x_for_y should work correctly', () => {
    const { alignment, Lx, Ly } = testData.valid_case;
    const pa = new PairwiseAlignment(alignment, Lx, Ly);
    alignment.forEach(([x, y]) => {
      expect(pa.get_y_for_x(x)).toBe(y);
      expect(pa.get_x_for_y(y)).toBe(x);
    });
  });

  test('Insert should maintain order and uniqueness', () => {
    const { alignment, Lx, Ly } = testData.insert_case;
    const pa = new PairwiseAlignment(alignment, Lx, Ly);
    expect(() => pa.insert([2, 3])).not.toThrow();
    expect(() => pa.insert([2, 4])).toThrow(); // Invalid insert (y clash)
    expect(() => pa.insert([6, 2])).toThrow(); // Invalid insert (order violation)
  });

  test('Transpose should swap x and y values', () => {
    const { alignment, Lx, Ly } = testData.valid_case;
    const pa = new PairwiseAlignment(alignment, Lx, Ly);
    const transposed = pa.transpose();
    expect(transposed).toEqual(new PairwiseAlignment(alignment.map(([x, y]) => [y, x]), Ly, Lx));
  });

  test('Composition should align transitive pairs correctly', () => {
    const { xy_alignment, yz_alignment, xz_alignment, Lx, Ly, Lz } = testData.composition_case;
    const xy = new PairwiseAlignment(xy_alignment, Lx, Ly);
    const yz = new PairwiseAlignment(yz_alignment, Ly, Lz);
    const xz = xy.compose(yz);
    expect(xz).toEqual(new PairwiseAlignment(xz_alignment, Lx, Lz)); // Expected result
  });
});
