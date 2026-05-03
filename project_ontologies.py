import os
import glob
import jpype
import importlib.util

# Get mowl jar path without importing mowl (JVM not started yet)
mowl_spec = importlib.util.find_spec("mowl")
mowl_path = os.path.dirname(mowl_spec.origin)
mowl_jars_dir = os.path.join(mowl_path, "lib")
mowl_jars = glob.glob(os.path.join(mowl_jars_dir, "*.jar"))

if not mowl_jars:
    raise FileNotFoundError(f"Could not find mOWL jars in {mowl_jars_dir}")

my_custom_jars = ["/home/zhapacfp/Git/multihop-gda/build/MultiHopProjector.jar"]
full_classpath = mowl_jars + my_custom_jars

jpype.startJVM(
    jpype.getDefaultJVMPath(),
    "-ea",
    "-Xmx4g",
    classpath=full_classpath,
    convertStrings=False
)

import jpype.imports  # enables "from java.xxx import ..." syntax

import mowl
mowl.init_jvm("4g")

from org.mowl.Projectors import MultiHopProjector
from mowl.datasets import PathDataset
from java.util import ArrayList

# ── ID maps ────────────────────────────────────────────────────────────────────

def load_id_map(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping

if not os.path.exists("data/entity_to_id.txt"):
    raise Exception(f"Entity mapping not found. Run 'owl_signature_to_ids.py'")

if not os.path.exists("data/relation_to_id.txt"):
    raise Exception(f"Relation mapping not found. Run 'owl_signature_to_ids.py'")

entity_to_id   = load_id_map("data/entity_to_id.txt")
relation_to_id = load_id_map("data/relation_to_id.txt")

# ── Query serialization ─────────────────────────────────────────────────────────

def serialize_query(expr):
    """Walk a Java QueryExpression and produce a string with IDs instead of IRIs."""
    name = str(expr.getClass().getSimpleName())
    if name == "Anchor":
        iri = str(expr.iri())
        return str(entity_to_id[iri])
    elif name == "Projection":
        rel_id  = relation_to_id[str(expr.relation())]
        child   = serialize_query(expr.child())
        return f"P({rel_id},{child})"
    elif name == "Intersection":
        children = ",".join(serialize_query(c) for c in expr.children().toArray())
        return f"I({children})"
    else:
        raise ValueError(f"Unknown QueryExpression type: {name}")

# ── Projection ──────────────────────────────────────────────────────────────────

with open("data/prefixes_to_ignore.txt", "r") as f:
    prefixes_to_ignore = [line.strip() for line in f if line.strip()]
print("Prefixes to ignore:", prefixes_to_ignore)
print("----")

prefixes_to_ignore = ArrayList(prefixes_to_ignore)

ontologies = [
    ("MP",      PathDataset("data/mp.owl").ontology),
    ("GO-Plus", PathDataset("data/go-plus.owl").ontology),
    ("HP",      PathDataset("data/hp.owl").ontology),
    ("Uberon",  PathDataset("data/uberon.owl").ontology),
    ("PATO",    PathDataset("data/pato.owl").ontology),
    ("UPheno",  PathDataset("data/upheno.owl").ontology),
]

projector = MultiHopProjector()

# Merged accumulators:
#   query_str (queryGraph.toString) → set of answer IRIs
#   query_str                       → representative Java MultihopQuery object
#   query_str                       → pattern string
merged_query_to_answers  = {}   # str → set[str]
query_str_to_java_query  = {}   # str → MultihopQuery (Java)
query_str_to_pattern     = {}   # str → str
all_disjoint_pairs = set()  # set of (str, str) pairs
for name, ontology in ontologies:
    print(f"\nProjecting {name}...")
    result      = projector.project(ontology, prefixes_to_ignore)
    pattern_map = result._1()   # HashMap[String, List[MultihopQuery]]
    query_map   = result._2()   # HashMap[String, List[String]]
    disjoint_pairs = result._3() # List[Pair[String, String]]
    # pattern_map: keep representative Java query object + pattern label per query_str
    for entry in pattern_map.entrySet():
        pattern = str(entry.getKey())
        for q in entry.getValue():
            query_str = str(q.queryGraph().toString())
            query_str_to_java_query.setdefault(query_str, q)
            query_str_to_pattern.setdefault(query_str, pattern)

    # query_map: merge answer IRIs
    for entry in query_map.entrySet():
        query_str = str(entry.getKey())
        if query_str not in merged_query_to_answers:
            merged_query_to_answers[query_str] = set()
        for ans in entry.getValue():
            merged_query_to_answers[query_str].add(str(ans))


    for pair in disjoint_pairs:
        all_disjoint_pairs.add((str(pair[0]), str(pair[1])))

            
print(f"\nTotal unique queries across all ontologies: {len(merged_query_to_answers)}")

# ── Serialize and write ─────────────────────────────────────────────────────────

skipped = 0

# query_to_answers.txt  →  <serialized_query_with_ids> TAB <answer_id> ...
# pattern_to_queries.txt →  <pattern> TAB <serialized_query_with_ids>

query_lines   = []   # (serialized_query, sorted answer ids)
pattern_lines = []   # (pattern, serialized_query)

for query_str, answer_iris in merged_query_to_answers.items():
    java_q  = query_str_to_java_query[query_str]
    pattern = query_str_to_pattern[query_str]

    try:
        serialized = serialize_query(java_q.queryGraph())
    except KeyError as e:
        skipped += 1
        continue

    answer_ids = sorted(
        entity_to_id[iri]
        for iri in answer_iris
        if iri in entity_to_id
    )
    if not answer_ids:
        skipped += 1
        continue

    query_lines.append((serialized, answer_ids))
    pattern_lines.append((pattern, serialized))

with open("data/query_to_answers.txt", "w") as f:
    for serialized, answer_ids in query_lines:
        f.write(serialized + "\t" + " ".join(map(str, answer_ids)) + "\n")

with open("data/pattern_to_queries.txt", "w") as f:
    for pattern, serialized in sorted(pattern_lines):
        f.write(pattern + "\t" + serialized + "\n")

with open("data/disjoint_pairs.txt", "w") as f: 
    for pair in all_disjoint_pairs: 
          f.write(f"{pair[0]}\t{pair[1]}\n")   
    

        
print(f"Written {len(query_lines)} queries ({skipped} skipped due to missing IDs)")
print(f"  data/query_to_answers.txt")
print(f"  data/pattern_to_queries.txt")
print(f"  data/disjoint_pairs.txt")
