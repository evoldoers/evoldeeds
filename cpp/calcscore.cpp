// calcscore.cpp
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <iterator>
#include <algorithm>
#include <cstring>   // strlen, memcmp

#include "sajson.h"

static inline bool isDigit(char c){ return c>='0' && c<='9'; }
static void die(const std::string& msg){ throw std::runtime_error(msg); }

struct Node {
    std::string id;
    std::string cigar;
    double distance{0.0};
    bool hasSeq{false};
    std::string seq;
    std::vector<int> children;
};

struct ModelParams {
    std::string alphabet;
    double seqlen{0.0}, inslen{0.0}, reslife{0.0};
    std::vector<double> root, evals;
    std::vector<std::vector<double>> evecs_l, evecs_r;
};

struct Expanded {
    std::vector<Node> nodes;             // preorder
    std::vector<int> parentIndex;        // -1 for root
    std::vector<double> distanceToParent;
    std::vector<std::vector<int>> childIndex;

    std::vector<std::string> alignment;
    std::vector<std::string> expandedCigar;
    std::vector<int> rootByColumn;
    int nRows{0};
    int nColumns{0};
    char gap{'-'};
    char wildcard{'*'};
};

// ---- helpers ----

static std::string expandCigarString(const std::string& cigar){
    std::string out;
    size_t i=0, n=cigar.size();
    while(i<n){
        if(!isDigit(cigar[i])) die("Malformed CIGAR (count expected)");
        size_t j=i;
        while(j<n && isDigit(cigar[j])) ++j;
        if(j==n) die("Malformed CIGAR (op missing)");
        int count = std::stoi(cigar.substr(i, j-i));
        char op = cigar[j];
        if(op!='M' && op!='I' && op!='D') die("Malformed CIGAR (op must be M/I/D)");
        out.append(count, op);
        i = j+1;
    }
    return out;
}

static bool key_equals(const sajson::string& k, const char* key){
    const size_t n = std::strlen(key);
    return k.length()==n && std::memcmp(k.data(), key, n)==0;
}

static bool find_field(const sajson::value& obj, const char* key, size_t& idx_out){
    if (obj.get_type() != sajson::TYPE_OBJECT) return false;
    const size_t n = obj.get_length();
    for (size_t i=0;i<n;++i){
        if (key_equals(obj.get_object_key(i), key)){
            idx_out = i;
            return true;
        }
    }
    return false;
}

static double to_number(const sajson::value& v){
    const auto t = v.get_type();
    if (t == sajson::TYPE_INTEGER) return static_cast<double>(v.get_integer_value());
    if (t == sajson::TYPE_DOUBLE)  return v.get_double_value();
    die("number expected");
    return 0.0;
}

// ---- tree build & expansion ----

static int buildTreeFromJson(const sajson::value& v, int parent,
                             std::vector<Node>& nodes,
                             std::vector<int>& parentIndex,
                             std::vector<double>& distanceToParent,
                             std::vector<std::vector<int>>& childIndex)
{
    if (v.get_type()!=sajson::TYPE_OBJECT) die("cigartree node is not an object");

    Node node;

    // id
    if (size_t idx; find_field(v, "id", idx)){
        sajson::value idv = v.get_object_value(idx);
        if (idv.get_type()!=sajson::TYPE_STRING) die("id must be string");
        node.id.assign(idv.as_string());
    }
    // cigar
    if (size_t idx; find_field(v, "cigar", idx)){
        sajson::value cv = v.get_object_value(idx);
        if (cv.get_type()!=sajson::TYPE_STRING) die("cigar must be string");
        node.cigar.assign(cv.as_string());
        if (node.cigar.empty()) die("cigar is empty");
    } else {
        die("cigar is missing");
    }
    // distance
    if (size_t idx; find_field(v, "distance", idx)){
        sajson::value dv = v.get_object_value(idx);
        const auto t = dv.get_type();
        if (t!=sajson::TYPE_INTEGER && t!=sajson::TYPE_DOUBLE) die("distance must be number");
        node.distance = to_number(dv);
    }
    // seq
    if (size_t idx; find_field(v, "seq", idx)){
        sajson::value sv = v.get_object_value(idx);
        if (sv.get_type()!=sajson::TYPE_STRING) die("seq must be string");
        node.seq.assign(sv.as_string());
        node.hasSeq = true;
    }

    int idxSelf = (int)nodes.size();
    nodes.push_back(node);
    parentIndex.push_back(parent);
    distanceToParent.push_back(node.distance);
    childIndex.emplace_back();

    if (size_t idx; find_field(v, "child", idx)){
        sajson::value ch = v.get_object_value(idx);
        if (ch.get_type()!=sajson::TYPE_ARRAY) die("child must be array");
        const auto L = ch.get_length();
        for (size_t i=0;i<L;++i){
            int ci = buildTreeFromJson(ch.get_array_element(i), idxSelf, nodes, parentIndex, distanceToParent, childIndex);
            childIndex[idxSelf].push_back(ci);
        }
    }

    if (nodes[idxSelf].cigar.empty()) die("Missing CIGAR on a node");
    return idxSelf;
}

static void advanceCursor(int row,
                          const std::vector<Node>& nodes,
                          const std::vector<std::string>& expCigars,
                          std::vector<size_t>& nextCigarPos,
                          std::vector<size_t>& nextSeqPos,
                          std::vector<std::string>& alignment,
                          const std::vector<std::vector<int>>& childIndex,
                          std::vector<int>& leaves,
                          std::vector<int>& internals,
                          std::vector<std::vector<int>>& branches,
                          bool isDelete,
                          char gap,
                          char wildcard)
{
    nextCigarPos[row]++;
    if (!isDelete){
        if (nodes[row].hasSeq){
            if (nextSeqPos[row] >= nodes[row].seq.size())
                die("Sequence ended prematurely at node " + std::to_string(row));
            char c = nodes[row].seq[nextSeqPos[row]];
            if (c==gap) die("Gap character found in sequence at node " + std::to_string(row));
            alignment[row].push_back(c);
            nextSeqPos[row]++;
        } else {
            alignment[row].push_back(wildcard);
        }

        if (childIndex[row].empty()){
            leaves.push_back(row);
        } else {
            for (int child : childIndex[row]){
                if (nextCigarPos[child] >= expCigars[child].size())
                    die("Child CIGAR ended early at node " + std::to_string(child));
                char childOp = expCigars[child][nextCigarPos[child]];
                if (childOp=='I') die("Insertion in child when M/D expected at node " + std::to_string(child));
                bool childIsDelete = (childOp=='D');
                advanceCursor(child, nodes, expCigars, nextCigarPos, nextSeqPos, alignment,
                              childIndex, leaves, internals, branches, childIsDelete, gap, wildcard);
                if (!childIsDelete) branches[row].push_back(child);
            }
            internals.push_back(row);
        }
    }
}

static Expanded expandCigarTreeFromStructures(const std::vector<Node>& nodesIn,
                                              const std::vector<int>& parentIndexIn,
                                              const std::vector<double>& distIn,
                                              const std::vector<std::vector<int>>& childIndexIn)
{
    Expanded E;
    E.nodes = nodesIn;
    E.parentIndex = parentIndexIn;
    E.distanceToParent = distIn;
    E.childIndex = childIndexIn;
    E.nRows = (int)nodesIn.size();
    E.expandedCigar.resize(E.nRows);
    E.alignment.assign(E.nRows, "");

    for (int i=0;i<E.nRows;++i) E.expandedCigar[i] = expandCigarString(E.nodes[i].cigar);

    std::vector<size_t> nextCigarPos(E.nRows, 0), nextSeqPos(E.nRows, 0);

    while (true){
        int nextInsertRow = -1;
        for (int r=E.nRows-1; r>=0; --r){
            if (nextCigarPos[r] < E.expandedCigar[r].size() && E.expandedCigar[r][nextCigarPos[r]]=='I'){
                nextInsertRow = r; break;
            }
        }
        if (nextInsertRow<0) break;

        std::vector<int> leaves, internals;
        std::vector<std::vector<int>> branches(E.nRows);
        advanceCursor(nextInsertRow, E.nodes, E.expandedCigar, nextCigarPos, nextSeqPos,
                      E.alignment, E.childIndex, leaves, internals, branches, false, E.gap, E.wildcard);

        E.rootByColumn.push_back(nextInsertRow);
        E.nColumns++;

        for (int r=0;r<E.nRows;++r){
            if ((int)E.alignment[r].size() < E.nColumns) E.alignment[r].push_back(E.gap);
        }
    }

    for (int r=0;r<E.nRows;++r){
        if (nextCigarPos[r] != E.expandedCigar[r].size())
            die("CIGAR not fully consumed at node " + std::to_string(r));
        if (E.nodes[r].hasSeq && nextSeqPos[r] != E.nodes[r].seq.size())
            die("Sequence not fully consumed at node " + std::to_string(r));
    }
    return E;
}

// ---- JSON printing ----

static void printString(const std::string& s, std::ostream& os){
    os << '"';
    for(char c: s){
        switch(c){
            case '\\': os << "\\\\"; break;
            case '"':  os << "\\\""; break;
            case '\b': os << "\\b";  break;
            case '\f': os << "\\f";  break;
            case '\n': os << "\\n";  break;
            case '\r': os << "\\r";  break;
            case '\t': os << "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c << std::dec;
                } else os << c;
        }
    }
    os << '"';
}
template<typename T>
static void printVector(const std::vector<T>& v, std::ostream& os){
    os << '[';
    for (size_t i=0;i<v.size();++i){ if (i) os << ','; os << v[i]; }
    os << ']';
}
static void printStringVector(const std::vector<std::string>& v, std::ostream& os){
    os << '[';
    for(size_t i=0;i<v.size();++i){ if(i) os << ','; printString(v[i], os); }
    os << ']';
}
static void printVectorVectorD(const std::vector<std::vector<double>>& M, std::ostream& os){
    os << '[';
    for(size_t i=0;i<M.size();++i){
        if(i) os << ',';
        os << '[';
        for(size_t j=0;j<M[i].size();++j){ if(j) os << ','; os << std::setprecision(17) << M[i][j]; }
        os << ']';
    }
    os << ']';
}
static void printVectorVectorI(const std::vector<std::vector<int>>& M, std::ostream& os){
    os << '[';
    for(size_t i=0;i<M.size();++i){
        if(i) os << ',';
        os << '[';
        for(size_t j=0;j<M[i].size();++j){ if(j) os << ','; os << M[i][j]; }
        os << ']';
    }
    os << ']';
}

// ---- params parsing ----

static ModelParams parseParams(const sajson::value& v){
    if (v.get_type()!=sajson::TYPE_OBJECT) die("\"params\" must be object");
    ModelParams P;

    if (size_t idx; find_field(v,"alphabet", idx)){
        sajson::value a = v.get_object_value(idx);
        if (a.get_type()!=sajson::TYPE_STRING) die("alphabet must be string");
        P.alphabet.assign(a.as_string(), a.get_string_length());
    }
    if (size_t idx; find_field(v,"seqlen", idx))  { sajson::value x = v.get_object_value(idx); P.seqlen  = to_number(x); }
    if (size_t idx; find_field(v,"inslen", idx))  { sajson::value x = v.get_object_value(idx); P.inslen  = to_number(x); }
    if (size_t idx; find_field(v,"reslife", idx)) { sajson::value x = v.get_object_value(idx); P.reslife = to_number(x); }

    if (size_t idx; find_field(v,"root", idx)){
        sajson::value r = v.get_object_value(idx);
        if (r.get_type()!=sajson::TYPE_ARRAY) die("root must be array");
        for (size_t i=0;i<r.get_length();++i) P.root.push_back(to_number(r.get_array_element(i)));
    }
    if (size_t idx; find_field(v,"evals", idx)){
        sajson::value e = v.get_object_value(idx);
        if (e.get_type()!=sajson::TYPE_ARRAY) die("evals must be array");
        for (size_t i=0;i<e.get_length();++i) P.evals.push_back(to_number(e.get_array_element(i)));
    }

    auto parse_matrix = [](const sajson::value& arr){
        if (arr.get_type()!=sajson::TYPE_ARRAY) die("matrix must be array");
        std::vector<std::vector<double>> M;
        for (size_t i=0;i<arr.get_length();++i){
            const auto &row = arr.get_array_element(i);
            if (row.get_type()!=sajson::TYPE_ARRAY) die("matrix rows must be arrays");
            std::vector<double> r;
            r.reserve(row.get_length());
            for (size_t j=0;j<row.get_length();++j){
                const auto &x = row.get_array_element(j);
                r.push_back(to_number(x));
            }
            M.push_back(std::move(r));
        }
        return M;
    };

    if (size_t idx; find_field(v,"evecs_l", idx)) { sajson::value L = v.get_object_value(idx); P.evecs_l = parse_matrix(L); }
    if (size_t idx; find_field(v,"evecs_r", idx)) { sajson::value R = v.get_object_value(idx); P.evecs_r = parse_matrix(R); }

    return P;
}

// ---- main ----

int main(){
    try{
        // read stdin
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::string input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());

        // sajson parse
        std::vector<char> buf(input.begin(), input.end());
        buf.push_back('\0');
        sajson::document doc = sajson::parse(
            sajson::dynamic_allocation(),
            sajson::mutable_string_view(buf.size()-1, buf.data())
        );
        if (!doc.is_valid()) die(std::string("JSON parse error: ") + doc.get_error_message_as_string());

        const sajson::value root = doc.get_root();
        if (root.get_type()!=sajson::TYPE_OBJECT) die("Top-level JSON must be an object");

        size_t idxTree, idxParams;
        if (!find_field(root, "cigartree", idxTree)) die("Missing top-level \"cigartree\"");
        if (!find_field(root, "params",    idxParams)) die("Missing top-level \"params\"");
        const sajson::value cigartreeJV = root.get_object_value(idxTree);
        const sajson::value paramsJV    = root.get_object_value(idxParams);

        // Build tree & expand
        std::vector<Node> nodes;
        std::vector<int> parentIndex;
        std::vector<double> distanceToParent;
        std::vector<std::vector<int>> childIndex;
        buildTreeFromJson(cigartreeJV, -1, nodes, parentIndex, distanceToParent, childIndex);

        Expanded E = expandCigarTreeFromStructures(nodes, parentIndex, distanceToParent, childIndex);

        // Params
        ModelParams P = parseParams(paramsJV);

        // Emit combined JSON (from internal state only)
        std::ostringstream os;
        os << '{';

        os << "\"alignment\":";      printStringVector(E.alignment, os);
        os << ",\"gap\":\"" << E.gap << "\"";
        os << ",\"wildcard\":\"" << E.wildcard << "\"";
        os << ",\"expandedCigar\":"; printStringVector(E.expandedCigar, os);
        os << ",\"nRows\":" << E.nRows;
        os << ",\"nColumns\":" << E.nColumns;
        os << ",\"rootByColumn\":";  printVector(E.rootByColumn, os);

        os << ",\"parentIndex\":";   printVector(E.parentIndex, os);

        os << ",\"distanceToParent\":[";
        for (size_t i=0;i<E.distanceToParent.size();++i){
            if(i) os << ',';
            os << std::setprecision(17) << E.distanceToParent[i];
        }
        os << ']';

        os << ",\"childIndex\":";    printVectorVectorI(E.childIndex, os);

        os << ",\"nodeMeta\":[";
        for (size_t i=0;i<E.nodes.size();++i){
            if (i) os << ',';
            os << '{';
            os << "\"id\":";
            if (E.nodes[i].id.empty()) os << "null"; else printString(E.nodes[i].id, os);
            os << ",\"hasSeq\":" << (E.nodes[i].hasSeq ? "true":"false");
            os << '}';
        }
        os << ']';

        os << ",\"params\":{";
        os << "\"alphabet\":"; printString(P.alphabet, os);
        os << ",\"seqlen\":"  << std::setprecision(17) << P.seqlen;
        os << ",\"inslen\":"  << std::setprecision(17) << P.inslen;
        os << ",\"reslife\":" << std::setprecision(17) << P.reslife;

        os << ",\"root\":[";
        for (size_t i=0;i<P.root.size();++i){ if(i) os<<','; os<<std::setprecision(17)<<P.root[i]; }
        os << "],\"evals\":[";
        for (size_t i=0;i<P.evals.size();++i){ if(i) os<<','; os<<std::setprecision(17)<<P.evals[i]; }
        os << "],\"evecs_l\":"; printVectorVectorD(P.evecs_l, os);
        os << ",\"evecs_r\":"; printVectorVectorD(P.evecs_r, os);
        os << '}';

        os << "}\n";
        std::cout << os.str();
        return 0;
    } catch (const std::exception& e){
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
