"""Project UPheno + annotation axioms into the KG edge list used downstream.

Runs OWL2VecStarGDAProjector against data/upheno.owl and writes
data/upheno_edges_gda.tsv. The projector emits standard SubClassOf and
equivalence triples plus phenotype to GO/UBERON edges extracted from
nested ObjectSomeValuesFrom axioms (HP and MP phenotypes both contribute).
"""

import os
import glob
import importlib.util
import jpype

mowl_spec = importlib.util.find_spec("mowl")
if mowl_spec is None or mowl_spec.origin is None:
    raise FileNotFoundError("mowl package not found in the active environment")
mowl_path = os.path.dirname(mowl_spec.origin)
mowl_jars = glob.glob(os.path.join(mowl_path, "lib", "*.jar"))
if not mowl_jars:
    raise FileNotFoundError(f"No mOWL jars under {mowl_path}/lib")

CUSTOM_JAR = "build/OWL2VecStarGDAProjector.jar"
if not os.path.exists(CUSTOM_JAR):
    raise FileNotFoundError(
        f"{CUSTOM_JAR} not found. Run ./compile_projector.sh first."
    )

jpype.startJVM(
    jpype.getDefaultJVMPath(),
    "-ea",
    "-Xmx8g",
    classpath=mowl_jars + [CUSTOM_JAR],
    convertStrings=False,
)
import jpype.imports  # noqa: F401  enables `from java.xxx import ...`

import mowl  # noqa: E402
mowl.init_jvm("8g")

from org.mowl.Projectors import OWL2VecStarGDAProjector  # noqa: E402
from mowl.datasets import PathDataset  # noqa: E402
from mowl.projection import Edge  # noqa: E402

OUT_PATH = "data/upheno_edges_gda.tsv"

ds = PathDataset("data/upheno.owl")
projector = OWL2VecStarGDAProjector(True, False, False)
raw_edges = projector.project(ds.ontology)

edges = [
    Edge(str(e.src()), str(e.rel()), str(e.dst()))
    for e in raw_edges
    if str(e.dst()) != ""
]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    for e in edges:
        f.write(f"{e.src}\t{e.rel}\t{e.dst}\n")

print(f"Wrote {len(edges)} edges to {OUT_PATH}")
