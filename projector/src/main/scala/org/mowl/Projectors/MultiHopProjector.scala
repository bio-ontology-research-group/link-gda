package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.owlapi.util._
import org.semanticweb.owlapi.search._
// Java imports
import java.util
import scala.collection.mutable.ListBuffer
import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._

class MultiHopProjector() extends AbstractProjector{

  def project(ontology: OWLOntology, prefixesToIgnore: java.util.List[String]) = {
    val prefixes_to_ignore = prefixesToIgnore.asScala.toList

    val tboxAxioms = ontology.getTBoxAxioms(imports).asScala.toList

    var subclassOfAxioms   = ListBuffer[OWLSubClassOfAxiom]()
    var equivalenceAxioms  = ListBuffer[OWLEquivalentClassesAxiom]()
    var disjointAxioms     = ListBuffer[OWLDisjointClassesAxiom]()
    var ignoredAxioms      = ListBuffer[OWLAxiom]()

    for (axiom <- tboxAxioms) {
      axiom.getAxiomType.getName match {
        case "SubClassOf"        => subclassOfAxioms  += axiom.asInstanceOf[OWLSubClassOfAxiom]
        case "EquivalentClasses" => equivalenceAxioms += axiom.asInstanceOf[OWLEquivalentClassesAxiom]
        case "DisjointClasses"   => disjointAxioms    += axiom.asInstanceOf[OWLDisjointClassesAxiom]
        case _                   => ignoredAxioms      += axiom
      }
    }

    // --- Filter axioms containing entities with ignored prefixes ---
    val filteredSubclassAxioms    = subclassOfAxioms.filterNot(ax => hasIgnoredPrefix(ax, prefixes_to_ignore))
    val filteredEquivalenceAxioms = equivalenceAxioms.filterNot(ax => hasIgnoredPrefix(ax, prefixes_to_ignore))
    val filteredDisjointAxioms    = disjointAxioms.filterNot(ax => hasIgnoredPrefix(ax, prefixes_to_ignore))

    // --- SubClassOf: answer = subclass, query = superclass expression ---
    //   Both atomic:      query = Anchor(superclass), answer = subclass
    //   Complex superclass: query = parseExpression(superclass), answer = subclass
    val subclassQueries = filteredSubclassAxioms.flatMap { ax =>
      val sub   = ax.getSubClass
      val sup   = ax.getSuperClass
      if (sub.isAnonymous) None
      else if (!sup.isAnonymous) {
        val answer = sub.asInstanceOf[OWLClass].getIRI.toString
        val query  = Anchor(sup.asInstanceOf[OWLClass].getIRI.toString)
        Some(MultihopQuery(answer, query))
      } else {
        val answer = sub.asInstanceOf[OWLClass].getIRI.toString
        scala.util.Try(OwlToQueryParser.parseExpression(sup))
          .toOption
          .map(expr => MultihopQuery(answer, expr))
      }
    }

    // --- EquivalentClasses: answers = all atomic classes, queries = all complex expressions ---
    val equivalenceQueries = filteredEquivalenceAxioms.flatMap { ax =>
      val parts = ax.getClassExpressions.asScala.toList
      val answers  = parts.filter(!_.isAnonymous).map(_.asInstanceOf[OWLClass].getIRI.toString)
      val complexes = parts.filter(_.isAnonymous)

      for {
        answer <- answers
        expr   <- complexes
        parsed <- scala.util.Try(OwlToQueryParser.parseExpression(expr)).toOption
      } yield MultihopQuery(answer, parsed)
    }

    val allQueries = subclassQueries ++ equivalenceQueries

    // --- Build result dictionaries ---
    val patternToQueries = scala.collection.mutable.Map[String, ListBuffer[MultihopQuery]]()
    val queryToAnswers   = scala.collection.mutable.Map[String, ListBuffer[String]]()

    allQueries.foreach { q =>
      val pattern  = describePattern(q.queryGraph)
      val queryKey = q.queryGraph.toString
      patternToQueries.getOrElseUpdate(pattern,  ListBuffer()) += q
      queryToAnswers  .getOrElseUpdate(queryKey, ListBuffer()) += q.answerIRI
    }

    // --- Disjoint pairs: only atomic OWLClass members, all pairwise combinations ---
    val disjointPairs = ListBuffer[java.util.List[String]]()
    for (ax <- filteredDisjointAxioms) {
      val atomicIRIs = ax.getClassExpressions.asScala.toList
        .filter(!_.isAnonymous)
        .map(_.asInstanceOf[OWLClass].getIRI.toString)
      for {
        i <- atomicIRIs.indices
        j <- (i + 1) until atomicIRIs.size
      } {
        val pair = new java.util.ArrayList[String]()
        pair.add(atomicIRIs(i))
        pair.add(atomicIRIs(j))
        disjointPairs += pair
      }
    }

    // --- Report ---
    println("MultiHop projection done!")
    println("\tSubClassOf axioms       : " + subclassOfAxioms.size + " (" + (subclassOfAxioms.size - filteredSubclassAxioms.size) + " prefix-filtered) -> " + subclassQueries.size + " queries")
    println("\tEquivalentClasses axioms: " + equivalenceAxioms.size + " (" + (equivalenceAxioms.size - filteredEquivalenceAxioms.size) + " prefix-filtered) -> " + equivalenceQueries.size + " queries")
    println("\tDisjointClasses axioms  : " + disjointAxioms.size + " (" + (disjointAxioms.size - filteredDisjointAxioms.size) + " prefix-filtered) -> " + disjointPairs.size + " pairs")
    println("\tIgnored axiom types     : " + ignoredAxioms.size)
    println("\tUnique queries          : " + queryToAnswers.size)
    println("\tQuery patterns:")
    patternToQueries.toSeq.sortBy(-_._2.size).foreach { case (pattern, queries) =>
      println("\t\t" + pattern + ": " + queries.size)
    }

    // --- Convert to Java maps ---
    val javaPatternToQueries = new java.util.HashMap[String, java.util.List[MultihopQuery]]()
    patternToQueries.foreach { case (k, v) => javaPatternToQueries.put(k, v.toList.asJava) }

    val javaQueryToAnswers = new java.util.HashMap[String, java.util.List[String]]()
    queryToAnswers.foreach { case (k, v) => javaQueryToAnswers.put(k, v.toList.asJava) }

    val javaDisjointPairs = new java.util.ArrayList[java.util.List[String]]()
    disjointPairs.foreach(javaDisjointPairs.add)

    (javaPatternToQueries, javaQueryToAnswers, javaDisjointPairs)
  }

  // Returns true if any entity in the axiom has an IRI starting with one of the given prefixes.
  private def hasIgnoredPrefix(axiom: OWLAxiom, prefixes: List[String]): Boolean =
    prefixes.nonEmpty && axiom.getClassesInSignature.asScala.exists { cls =>
      val iri = cls.getIRI.toString
      prefixes.exists(iri.startsWith)
    }

  
  def describePattern(expr: QueryExpression): String = expr match {
    case Anchor(_)              => "Anchor"
    case Projection(_, child)   => "P(" + describePattern(child) + ")"
    case Intersection(children) => "I(" + children.asScala.map(describePattern).mkString(",") + ")"
  }

  // Abstract method stubs required by AbstractProjector
  override def processOntClass(ontClass: OWLClass, ontology: OWLOntology): List[Triple] = Nil
  def project(ontology: OWLOntology, withIndividuals: Boolean, verbose: Boolean): java.util.List[Triple] = Nil.asJava
  def projectAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Triple] = Nil
  def projectAxiom(go_class: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple] = Nil
  def projectAxiom(axiom: OWLAxiom): List[org.mowl.Types.Triple] = Nil
  def projectAxiom(axiom: OWLClassAxiom): List[org.mowl.Types.Triple] = Nil
  def projectAxiom(axiom: OWLAxiom, with_individuals: Boolean, verbose: Boolean): List[Triple] = Nil
}
