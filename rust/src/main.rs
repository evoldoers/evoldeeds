use std::io::{self, Read};

use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
struct Node {
    id: Option<String>,
    cigar: String,
    distance: f64,
    has_seq: bool,
    seq: String,
    children: Vec<usize>, // filled after recursion
}

#[derive(Debug)]
struct ModelParams {
    alphabet: String,
    seqlen: f64,
    inslen: f64,
    reslife: f64,
    root: Vec<f64>,
    evals: Vec<f64>,
    evecs_l: Vec<Vec<f64>>,
    evecs_r: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct Expanded {
    // topology / metadata
    nodes: Vec<Node>,                 // preorder
    parent_index: Vec<i32>,           // -1 for root
    distance_to_parent: Vec<f64>,
    child_index: Vec<Vec<usize>>,

    // alignment
    alignment: Vec<String>,           // one row per node
    expanded_cigar: Vec<String>,
    root_by_column: Vec<usize>,
    n_rows: usize,
    n_cols: usize,

    gap: char,
    wildcard: char,
}

fn die<T>(msg: &str) -> Result<T, String> {
    Err(msg.to_string())
}

fn to_number(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| "number out of f64 range".to_string()),
        _ => die("number expected"),
    }
}

fn get<'a>(m: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    m.get(key)
}

fn as_obj<'a>(v: &'a Value) -> Result<&'a Map<String, Value>, String> {
    match v {
        Value::Object(m) => Ok(m),
        _ => die("object expected"),
    }
}

fn as_arr<'a>(v: &'a Value) -> Result<&'a Vec<Value>, String> {
    match v {
        Value::Array(a) => Ok(a),
        _ => die("array expected"),
    }
}

fn as_str<'a>(v: &'a Value) -> Result<&'a str, String> {
    match v {
        Value::String(s) => Ok(s),
        _ => die("string expected"),
    }
}

fn expand_cigar_string(cigar: &str) -> Result<String, String> {
    let bytes = cigar.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();
    let mut out = String::new();

    while i < n {
        if !(bytes[i].is_ascii_digit()) {
            return die("Malformed CIGAR (count expected)");
        }
        let start = i;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i >= n {
            return die("Malformed CIGAR (op missing)");
        }
        let count: usize = std::str::from_utf8(&bytes[start..i])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| "Malformed CIGAR (bad count)".to_string())?;
        let op = bytes[i] as char;
        if op != 'M' && op != 'I' && op != 'D' {
            return die("Malformed CIGAR (op must be M/I/D)");
        }
        out.extend(std::iter::repeat(op).take(count));
        i += 1;
    }
    Ok(out)
}

fn build_tree_from_json(
    v: &Value,
    parent: i32,
    nodes: &mut Vec<Node>,
    parent_index: &mut Vec<i32>,
    distance_to_parent: &mut Vec<f64>,
    child_index: &mut Vec<Vec<usize>>,
) -> Result<usize, String> {
    let m = as_obj(v)?;

    let mut node = Node {
        id: None,
        cigar: String::new(),
        distance: 0.0,
        has_seq: false,
        seq: String::new(),
        children: vec![],
    };
    let mut child_arr: Option<&Vec<Value>> = None;

    // scan fields once
    for (k, val) in m {
        match k.as_str() {
            "id" => {
                node.id = Some(as_str(val)?.to_string());
            }
            "cigar" => {
                node.cigar = as_str(val)?.to_string();
            }
            "distance" => {
                node.distance = to_number(val)?;
            }
            "seq" => {
                node.seq = as_str(val)?.to_string();
                node.has_seq = true;
            }
            "child" => {
                let arr = as_arr(val)?;
                child_arr = Some(arr);
            }
            _ => { /* ignore unknowns */ }
        }
    }

    if node.cigar.is_empty() {
        // help: list keys present
        let keys: Vec<_> = m.keys().cloned().collect();
        return Err(format!("Missing CIGAR on a node; fields present: {:?}", keys));
    }

    let idx_self = nodes.len();
    nodes.push(node);
    parent_index.push(parent);
    distance_to_parent.push(nodes[idx_self].distance);
    child_index.push(vec![]);

    if let Some(arr) = child_arr {
        for child_v in arr {
            let ci = build_tree_from_json(child_v, idx_self as i32, nodes, parent_index, distance_to_parent, child_index)?;
            child_index[idx_self].push(ci);
        }
    }
    Ok(idx_self)
}

fn advance_cursor(
    row: usize,
    nodes: &Vec<Node>,
    exp_cigars: &Vec<String>,
    next_cigar_pos: &mut Vec<usize>,
    next_seq_pos: &mut Vec<usize>,
    alignment: &mut Vec<String>,
    child_index: &Vec<Vec<usize>>,
    leaves: &mut Vec<usize>,
    internals: &mut Vec<usize>,
    branches: &mut Vec<Vec<usize>>,
    is_delete: bool,
    gap: char,
    wildcard: char,
) -> Result<(), String> {
    next_cigar_pos[row] += 1;

    if !is_delete {
        if nodes[row].has_seq {
            if next_seq_pos[row] >= nodes[row].seq.len() {
                return Err(format!("Sequence ended prematurely at node {}", row));
            }
            let c = nodes[row].seq.as_bytes()[next_seq_pos[row]] as char;
            if c == gap {
                return Err(format!("Gap character found in sequence at node {}", row));
            }
            alignment[row].push(c);
            next_seq_pos[row] += 1;
        } else {
            alignment[row].push(wildcard);
        }

        if child_index[row].is_empty() {
            leaves.push(row);
        } else {
            for &child in &child_index[row] {
                if next_cigar_pos[child] >= exp_cigars[child].len() {
                    return Err(format!("Child CIGAR ended early at node {}", child));
                }
                let child_op = exp_cigars[child].as_bytes()[next_cigar_pos[child]] as char;
                if child_op == 'I' {
                    return Err(format!(
                        "Insertion in child when M/D expected at node {}",
                        child
                    ));
                }
                let child_is_delete = child_op == 'D';
                advance_cursor(
                    child,
                    nodes,
                    exp_cigars,
                    next_cigar_pos,
                    next_seq_pos,
                    alignment,
                    child_index,
                    leaves,
                    internals,
                    branches,
                    child_is_delete,
                    gap,
                    wildcard,
                )?;
                if !child_is_delete {
                    branches[row].push(child);
                }
            }
            internals.push(row);
        }
    }
    Ok(())
}

fn expand_cigar_tree_from_structures(
    nodes_in: Vec<Node>,
    parent_index_in: Vec<i32>,
    dist_in: Vec<f64>,
    child_index_in: Vec<Vec<usize>>,
) -> Result<Expanded, String> {
    let mut e = Expanded {
        nodes: nodes_in,
        parent_index: parent_index_in,
        distance_to_parent: dist_in,
        child_index: child_index_in,
        alignment: vec![],
        expanded_cigar: vec![],
        root_by_column: vec![],
        n_rows: 0,
        n_cols: 0,
        gap: '-',
        wildcard: '*',
    };
    e.n_rows = e.nodes.len();
    e.expanded_cigar = Vec::with_capacity(e.n_rows);
    e.alignment = vec![String::new(); e.n_rows];

    for node in &e.nodes {
        e.expanded_cigar.push(expand_cigar_string(&node.cigar)?);
    }

    let mut next_cigar_pos = vec![0usize; e.n_rows];
    let mut next_seq_pos = vec![0usize; e.n_rows];

    loop {
        // highest-numbered row with next op == 'I'
        let mut next_insert_row: Option<usize> = None;
        for r in (0..e.n_rows).rev() {
            if next_cigar_pos[r] < e.expanded_cigar[r].len()
                && e.expanded_cigar[r].as_bytes()[next_cigar_pos[r]] as char == 'I'
            {
                next_insert_row = Some(r);
                break;
            }
        }
        let Some(root_row) = next_insert_row else { break; };

        let mut leaves = Vec::new();
        let mut internals = Vec::new();
        let mut branches = vec![Vec::<usize>::new(); e.n_rows];

        advance_cursor(
            root_row,
            &e.nodes,
            &e.expanded_cigar,
            &mut next_cigar_pos,
            &mut next_seq_pos,
            &mut e.alignment,
            &e.child_index,
            &mut leaves,
            &mut internals,
            &mut branches,
            false,
            e.gap,
            e.wildcard,
        )?;

        e.root_by_column.push(root_row);
        e.n_cols += 1;

        // pad rows
        for r in 0..e.n_rows {
            if e.alignment[r].len() < e.n_cols {
                e.alignment[r].push(e.gap);
            }
        }
    }

    // verify all consumed
    for r in 0..e.n_rows {
        if next_cigar_pos[r] != e.expanded_cigar[r].len() {
            return Err(format!(
                "CIGAR not fully consumed at node {} (pos {} of {})",
                r, next_cigar_pos[r], e.expanded_cigar[r].len()
            ));
        }
        if e.nodes[r].has_seq && next_seq_pos[r] != e.nodes[r].seq.len() {
            return Err(format!(
                "Sequence not fully consumed at node {} (pos {} of {})",
                r, next_seq_pos[r], e.nodes[r].seq.len()
            ));
        }
    }

    Ok(e)
}

fn parse_params(v: &Value) -> Result<ModelParams, String> {
    let m = as_obj(v)?;
    let alphabet = as_str(get(m, "alphabet").ok_or_else(|| "\"alphabet\" missing".to_string())?)?.to_string();

    let seqlen = to_number(get(m, "seqlen").ok_or_else(|| "\"seqlen\" missing".to_string())?)?;
    let inslen = to_number(get(m, "inslen").ok_or_else(|| "\"inslen\" missing".to_string())?)?;
    let reslife = to_number(get(m, "reslife").ok_or_else(|| "\"reslife\" missing".to_string())?)?;

    let root = as_arr(get(m, "root").ok_or_else(|| "\"root\" missing".to_string())?)?
        .iter().map(to_number).collect::<Result<Vec<_>,_>>()?;

    let evals = as_arr(get(m, "evals").ok_or_else(|| "\"evals\" missing".to_string())?)?
        .iter().map(to_number).collect::<Result<Vec<_>,_>>()?;

    let parse_matrix = |key: &str| -> Result<Vec<Vec<f64>>, String> {
        let arr = as_arr(get(m, key).ok_or_else(|| format!("\"{}\" missing", key))?)?;
        let mut out = Vec::with_capacity(arr.len());
        for row_v in arr {
            let row = as_arr(row_v)?
                .iter()
                .map(to_number)
                .collect::<Result<Vec<_>,_>>()?;
            out.push(row);
        }
        Ok(out)
    };

    let evecs_l = parse_matrix("evecs_l")?;
    let evecs_r = parse_matrix("evecs_r")?;

    Ok(ModelParams { alphabet, seqlen, inslen, reslife, root, evals, evecs_l, evecs_r })
}

fn main() -> Result<(), String> {
    // read stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).map_err(|e| e.to_string())?;

    let root_v: Value = serde_json::from_str(&input).map_err(|e| format!("JSON parse error: {}", e))?;
    let root = as_obj(&root_v)?;

    let cigartree_v = get(root, "cigartree").ok_or_else(|| "Missing top-level \"cigartree\"".to_string())?;
    let params_v    = get(root, "params").ok_or_else(|| "Missing top-level \"params\"".to_string())?;

    // build tree
    let mut nodes: Vec<Node> = Vec::new();
    let mut parent_index: Vec<i32> = Vec::new();
    let mut distance_to_parent: Vec<f64> = Vec::new();
    let mut child_index: Vec<Vec<usize>> = Vec::new();

    build_tree_from_json(cigartree_v, -1, &mut nodes, &mut parent_index, &mut distance_to_parent, &mut child_index)?;

    // expand
    let e = expand_cigar_tree_from_structures(nodes, parent_index, distance_to_parent, child_index)?;

    // params
    let p = parse_params(params_v)?;

    // Assemble JSON purely from internal state
    let node_meta = e.nodes.iter().map(|n| {
        json!({
            "id": n.id.clone(),
            "hasSeq": n.has_seq
        })
    }).collect::<Vec<_>>();

    let out = json!({
        "alignment": e.alignment,
        "gap": e.gap.to_string(),
        "wildcard": e.wildcard.to_string(),
        "expandedCigar": e.expanded_cigar,
        "nRows": e.n_rows,
        "nColumns": e.n_cols,
        "rootByColumn": e.root_by_column,
        "parentIndex": e.parent_index,
        "distanceToParent": e.distance_to_parent,
        "childIndex": e.child_index,
        "nodeMeta": node_meta,
        "params": {
            "alphabet": p.alphabet,
            "seqlen": p.seqlen,
            "inslen": p.inslen,
            "reslife": p.reslife,
            "root": p.root,
            "evals": p.evals,
            "evecs_l": p.evecs_l,
            "evecs_r": p.evecs_r
        }
    });

    // print compact JSON
    println!("{}", serde_json::to_string(&out).unwrap());
    Ok(())
}
