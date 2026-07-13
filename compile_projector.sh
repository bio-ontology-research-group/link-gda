set -e

scalac -cp "$(echo $HOME/miniforge3/envs/link-gda/lib/python3.11/site-packages/mowl/lib/*.jar | tr ' ' ':')" -d build projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala

jar cf ~/Git/link-gda/build/OWL2VecStarGDAProjector.jar -C ~/Git/link-gda/build .
