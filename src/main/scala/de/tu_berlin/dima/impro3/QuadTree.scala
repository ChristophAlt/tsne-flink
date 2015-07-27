/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.tu_berlin.dima.impro3

import java.io.Serializable

import breeze.linalg.{DenseVector, Vector, squaredDistance}

import scala.math._


class QuadTree(var parent: Option[QuadTree], val boundary: Cell)
  extends Serializable {

  var northWest, northEast, southWest, southEast: Option[QuadTree] = None
  var isLeaf = true
  var cumSize = 0
  var centerOfMass = DenseVector(0.0, 0.0)
  var sum = DenseVector(0.0, 0.0)
  var thisPoint: Option[Vector[Double]] = None

  def insert(point: Vector[Double]): Boolean = {

    if (boundary.contains(point)) {
      sum += point
      cumSize += 1

      centerOfMass = sum / cumSize.toDouble

      // is this a leaf node?
      if (isLeaf) {
        thisPoint match {
          // some point already assigned to this node
          case Some(leafPoint) =>
            // if it's the same point, we're good
            if (leafPoint.equals(point)) {
              true
            // otherwise, divide into subtree
            } else {
              subDivide()
              isLeaf = false
              // insert this leafs point and the current point into the subtree
              insertIntoSubTree(leafPoint)
              insertIntoSubTree(point)
              thisPoint = None
              true
            }
          // leaf but no point assigned, so lets assign one
          case _ =>
            thisPoint = Option(point)
            true
        }
      // not a leaf node, insert into subtree
      } else {
        insertIntoSubTree(point)
      }
    // not belonging to this tree
    } else {
      false
    }
  }

  def subDivide() = {
    val newWidth = 0.5 * boundary.hWidth
    val newHeight = 0.5 * boundary.hWidth

    val nW = QuadTree(Option(this), Cell(boundary.x - newWidth, boundary.y + newHeight, newWidth, newHeight))
    val nE = QuadTree(Option(this), Cell(boundary.x + newWidth, boundary.y + newHeight, newWidth, newHeight))
    val sW = QuadTree(Option(this), Cell(boundary.x - newWidth, boundary.y - newHeight, newWidth, newHeight))
    val sE = QuadTree(Option(this), Cell(boundary.x + newWidth, boundary.y - newHeight, newWidth, newHeight))

    northWest = Option(nW)
    northEast = Option(nE)
    southWest = Option(sW)
    southEast = Option(sE)
  }

  def insertIntoSubTree(point: Vector[Double]): Boolean = {
    var success = false
    if (!success) {
      success = checkAndInsert(point, northWest)
    }
    if (!success) {
      success = checkAndInsert(point, northEast)
    }
    if (!success) {
      success = checkAndInsert(point, southWest)
    }
    if (!success) {
      success = checkAndInsert(point, southEast)
    }
    success
  }

  private def checkAndInsert(point: Vector[Double], subTree: Option[QuadTree]): Boolean = {
    subTree match {
      case Some(tree) =>
        if (tree.boundary.contains(point)) {
          tree.insert(point)
        } else {
          false
        }
      case _ => false
    }
  }

  def computeRepulsiveForce(point: Vector[Double], theta: Double):
    (Vector[Double], Double) = {
    var sumQ = 0.0

    // in case the node is empty or it's the query point, ignore
    if ((isLeaf && cumSize == 0) || (isLeaf && thisPoint.get.equals(point))) {
      (DenseVector(0.0, 0.0), 0.0)
    } else {
      // if this tree is a leaf or the summary condition is satisfied, compute the repulsive force
      // and partial sum of Q
      val D = squaredDistance(point, centerOfMass)
      if (isLeaf || (max(boundary.hHeigth, boundary.hWidth) / D < theta)) {
        // Q = q_i,cell * Z = 1 / (1 + D(y_i, y_cell)^2)
        val Q = 1.0 / (1.0 + D)
        // account for Z_part = N_cell * q_i,cell * Z
        val mult = cumSize * Q
        sumQ += mult
        val repForce = mult * Q * (point - centerOfMass)
        (repForce, sumQ)
        // otherwise go to the subtree
      } else {
        // that's not good style
        val (nWF, nWSum) = northWest.get.computeRepulsiveForce(point, theta)
        val (nEF, nESum) = northEast.get.computeRepulsiveForce(point, theta)
        val (sWF, sWSum) = southWest.get.computeRepulsiveForce(point, theta)
        val (sEF, sESum) = southEast.get.computeRepulsiveForce(point, theta)
        (nWF + nEF + sWF + sEF, nWSum + nESum + sWSum + sESum)
      }
    }
  }
}

object QuadTree {
  val NUM_DIMENSIONS = 2
  val NODE_CAPACITY = 1

  def apply(parent: Option[QuadTree], boundary: Cell): QuadTree = {
    new QuadTree(parent, boundary)
  }
}
