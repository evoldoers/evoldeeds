import Ajv from "ajv";

const ajv = new Ajv({ strict: true });

export const cigarTreeSchema = {
  type: "object",
  required: ["cigar"],
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
      minItems: 2,
      maxItems: 2,
      items: { $ref: "#" }
    }
  },
  additionalProperties: false
};

const validateSchema = ajv.compile(cigarTreeSchema);

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
  isRoot = false,
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

  // Rule: id required on leaf nodes
  if (isLeaf) {
    if (node.id === undefined) {
      errors.push(`Missing required 'id' at leaf node ${path}`);
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

  if (node.child) {
    node.child.forEach((child, i) => {
      errors.push(...traverseTree(child, childLen, {
        seenIds,
        leafIds,
        idLengthMap,
        isRoot: false,
        path: `${path}.child[${i}]`
      }));
    });
  }

  return errors;
}

export function validateCigarTree(tree, options = {}) {
  const { throwOnError = false, seqById = null } = options;

  const schemaErrors = [];
  const logicErrors = [];
  const consistencyErrors = [];

  const validSchema = validateSchema(tree);
  if (!validSchema) {
    schemaErrors.push(
      ...(validateSchema.errors || []).map(e => `${e.instancePath} ${e.message}`)
    );
  } else {
    const seenIds = new Set();
    const leafIds = new Set();
    const idLengthMap = {};

    logicErrors.push(
      ...traverseTree(tree, 0, {
        seenIds,
        leafIds,
        idLengthMap,
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
