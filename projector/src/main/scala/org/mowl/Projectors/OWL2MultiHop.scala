package org.mowl.Projectors

import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model._
import collection.JavaConverters._

// --- 1. The Query Model (ADT) ---
sealed trait QueryExpression
// An Anchor is now a constant class found INSIDE the query body (RHS)
case class Anchor(iri: String) extends QueryExpression
case class Projection(relation: String, child: QueryExpression) extends QueryExpression
case class Intersection(children: java.util.List[QueryExpression]) extends QueryExpression

// A full Query definition: The "Answer" (LHS) and the "Graph" (RHS)
case class MultihopQuery(answerIRI: String, queryGraph: QueryExpression)

// --- 2. The Parser Logic ---
object OwlToQueryParser {

  // Recursively parses the RHS expression
  def parseExpression(expr: OWLClassExpression): QueryExpression = expr match {
    
    // CASE: Atomic Class -> It's an Anchor in the query graph
    case c: OWLClass => 
      Anchor(c.getIRI.toString)

    // CASE: Projection (ObjectSomeValuesFrom)
    case some: OWLObjectSomeValuesFrom =>
      val property = some.getProperty.asOWLObjectProperty.getIRI.toString
      Projection(property, parseExpression(some.getFiller))

    // CASE: Intersection (ObjectIntersectionOf)
    case and: OWLObjectIntersectionOf =>
      // Convert Java Stream to Scala List and map
      val operands = and.getOperands.asScala.toList.map(parseExpression)
      Intersection(operands.asJava)

    // Handle other cases safely
    case _ => throw new IllegalArgumentException(s"Unsupported expression type: $expr")
  }

  // Parses a full Axiom into our Query structure
  def parseAxiom(axiom: OWLEquivalentClassesAxiom): Option[MultihopQuery] = {
    // Split the equivalence into Named Class (LHS) and Complex Expression (RHS)
    val parts = axiom.getClassExpressions.asScala.toList
    
    val (lhsList, rhsList) = parts.partition(!_.isAnonymous)

    (lhsList.headOption, rhsList.headOption) match {
      case (Some(lhsClass), Some(rhsExpr)) =>
        val answer = lhsClass.asInstanceOf[OWLClass].getIRI.toString
        val graph = parseExpression(rhsExpr)
        Some(MultihopQuery(answer, graph))
      
      case _ => 
        println(s"Skipping axiom: Could not clearly separate LHS and RHS. $axiom")
        None
    }
  }
}

