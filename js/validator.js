import Ajv from "ajv";
import { secDependencies } from "mathjs";

const ajv = new Ajv({ strict: true });

// Schema for MSA + tree with no sequences (suitable for posting to a server that already has the sequences)
export const cigarTreeSchema = {
  type: "object",
  required: ["cigar"],
  properties: {
    id: { type: "string" },
    cigar: {
      type: "string",
      pattern: "^([0-9]+[MID])*$"
    },
    distance: {
      type: "number",
      minimum: 0
    },
    child: {
      type: "array",
      minItems: 2,
      maxItems: 2,
      items: { $ref: "#" }
    }
  },
  additionalProperties: false
};

// Schema for MSA + tree with sequences at leaves (suitable for standalone representation, likelihood calculations, ancestral reconstruction etc.)
const seqDecoratedCigarTreeSchema = {
  type: "object",
  required: ["cigar","child"],
  properties: {
    id: { type: "string" },
    cigar: {
      type: "string",
      pattern: "^([0-9]+I)?$"
    },
    child: {
      type: "array",
      minItems: 1,
      items: { $ref: "#/$defs/internalOrLeaf" }
    }
  },
  additionalProperties: false,
  $defs: {
    internalOrLeaf: {
      oneOf: [
        { $ref: "#/$defs/internalNode" },
        { $ref: "#/$defs/leafNode" }
      ]
    },
    internalNode: {
      type: "object",
      required: ["cigar", "distance", "child"],
      properties: {
        id: { type: "string" },
        cigar: {
          type: "string",
          pattern: "^([0-9]+[MID])+$"
        },
        distance: {
          type: "number",
          minimum: 0
        },
        child: {
          type: "array",
          minItems: 1,
          items: { $ref: "#/$defs/internalOrLeaf" }
        }
      },
      additionalProperties: false
    },
    leafNode: {
      type: "object",
      required: ["cigar", "distance", "seq"],
      properties: {
        id: { type: "string" },
        cigar: {
          type: "string",
          pattern: "^([0-9]+[MID])+$"
        },
        distance: {
          type: "number",
          minimum: 0
        },
        seq: {
          type: "string"
        }
      },
      additionalProperties: false
    }
  }
};


const validateSchema = ajv.compile(cigarTreeSchema);
const validateSeqDecoratedSchema = ajv.compile(seqDecoratedCigarTreeSchema);

function expandCigar(cigar) {
  const regex = /(\d+)([MID])/g;
  const result = [];
  let match;
  while ((match = regex.exec(cigar)) !== null) {
    result.push([parseInt(match[1], 10), match[2]]);
  }
  return result;
}

function getCigarLengths(cigar) {
  const ops = expandCigar(cigar);
  let parentLen = 0;
  let childLen = 0;
  for (const [len, op] of ops) {
    if (op === "M") {
      parentLen += len;
      childLen += len;
    } else if (op === "I") {
      childLen += len;
    } else if (op === "D") {
      parentLen += len;
    }
  }
  return { parentLen, childLen };
}

function traverseTree(node, expectedParentLen, {
  seenIds = new Set(),
  leafIds = new Set(),
  idLengthMap = {},
  idSeqMap = {},
  isRoot = false,
  seqDecorated = false,  // if the tree is sequence decorated then we don't require ids at leaf nodes
  path = "<root>"
} = {}) {
  const errors = [];

  const { parentLen, childLen } = getCigarLengths(node.cigar);
  if (parentLen !== expectedParentLen) {
    const label = node.id ?? path;
    console.warn(node.cigar, parentLen, expectedParentLen);
    errors.push(`Length mismatch at node ${label}: expected parent length ${expectedParentLen}, got ${parentLen}`);
  }

  const isLeaf = !node.child || node.child.length === 0;

  // Rule: distance required on all but root
  if (!isRoot && node.distance === undefined) {
    errors.push(`Missing required 'distance' at ${path}`);
  }

  if (node.seq !== undefined && childLen !== node.seq.length) {
    errors.push(`Sequence length mismatch at ${path}: expected ${childLen}, got ${node.seq.length}`);
  }

  // Rule: id required on leaf nodes, unless seqDecorated
  if (isLeaf) {
    if (node.id === undefined) {
      if (seqDecorated) {
        if (node.seq === undefined) {
          errors.push(`Missing required 'seq' at leaf node ${path}`);
        }
      } else {
        errors.push(`Missing required 'id' at leaf node ${path}`);
      }
    } else {
      if (seenIds.has(node.id)) {
        errors.push(`Duplicate ID '${node.id}' found at ${path}`);
      } else {
        seenIds.add(node.id);
      }
      leafIds.add(node.id);
      idLengthMap[node.id] = childLen;
    }
  } else if (node.id !== undefined) {
    if (seenIds.has(node.id)) {
      errors.push(`Duplicate ID '${node.id}' found at ${path}`);
    } else {
      seenIds.add(node.id);
    }
  }

  if (node.seq !== undefined && node.id !== undefined) {
    if (idSeqMap[node.id]) {
      errors.push(`Duplicate sequence for ID '${node.id}' at ${path}`);
    } else {
      idSeqMap[node.id] = node.seq;
    }
  }

  if (node.child) {
    node.child.forEach((child, i) => {
      errors.push(...traverseTree(child, childLen, {
        seenIds,
        leafIds,
        idLengthMap,
        idSeqMap,
        seqDecorated,
        isRoot: false,
        path: `${path}.child[${i}]`
      }));
    });
  }

  return errors;
}

export function validateCigarTree(tree, options = {}) {
  const { throwOnError = false, seqById = null, seqDecorated = false } = options;

  const schemaErrors = [];
  const logicErrors = [];
  const consistencyErrors = [];

  const validator = seqDecorated ? validateSeqDecoratedSchema : validateSchema;
  const validSchema = validator(tree);
  if (!validSchema) {
    schemaErrors.push(
      ...(validator.errors || []).map(e => `${e.instancePath} ${e.message}`)
    );
  } else {
    const seenIds = new Set();
    const leafIds = new Set();
    const idLengthMap = {};  // lengths computed from CIGAR strings
    const idSeqMap = {};   // sequences inside the CIGAR tree

    logicErrors.push(
      ...traverseTree(tree, 0, {
        seenIds,
        leafIds,
        idLengthMap,
        idSeqMap,
        seqDecorated,
        isRoot: true
      })
    );

    if (seqById) {
      const expectedIds = new Set(Object.keys(seqById));

      for (const id of leafIds) {
        if (!expectedIds.has(id)) {
          consistencyErrors.push(`Leaf node ID '${id}' not found in seqById`);
        }
      }
      for (const id of expectedIds) {
        if (!leafIds.has(id)) {
          consistencyErrors.push(`seqById ID '${id}' missing in leaf nodes`);
        }
      }
      for (const id of expectedIds) {
        if (leafIds.has(id)) {
          const seqLen = seqById[id].length;
          const cigarLen = idLengthMap[id];
          if (seqLen !== cigarLen) {
            consistencyErrors.push(`Leaf '${id}' has length ${cigarLen} from CIGAR but ${seqLen} from sequence`);
          }
          if (seqDecorated && idSeqMap[id] !== seqById[id]) {
            consistencyErrors.push(`Leaf '${id}' has different sequence in CIGAR tree and supplied seqById`);
          }
        }
      }
    }
  }

  const errors = [...schemaErrors, ...logicErrors, ...consistencyErrors];
  const valid = errors.length === 0;

  if (throwOnError && !valid) {
    throw new Error("CIGAR tree validation failed:\n" + allErrors.join("\n"));
  }

  return {
    valid,
    errors,
    schemaErrors,
    logicErrors,
    consistencyErrors
  };
}
