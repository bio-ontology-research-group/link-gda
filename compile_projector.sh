set -e

# scalac -cp "$(echo $HOME/miniforge3/envs/multihopgda/lib/python3.11/site-packages/mowl/lib/*.jar | tr ' ' ':')" -d build projector/src/main/scala/org/mowl/Projectors/OWL2MultiHop.scala projector/src/main/scala/org/mowl/Projectors/MultiHopProjector.scala

scalac -cp "$(echo $HOME/miniforge3/envs/multihopgda/lib/python3.11/site-packages/mowl/lib/*.jar | tr ' ' ':')" -d build projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala

# jar cf ~/Git/multihop-gda/build/MultiHopProjector.jar -C ~/Git/multihop-gda/build .
jar cf ~/Git/multihop-gda/build/OWL2VecStarGDAProjector.jar -C ~/Git/multihop-gda/build .
