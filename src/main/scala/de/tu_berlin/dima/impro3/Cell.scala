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

import breeze.linalg.Vector

// right now we only support 2D quadtrees
class Cell(var x: Double, var y: Double, var hWidth: Double, var hHeigth: Double)
  extends Serializable {

  def this(x: Double, y: Double, length: Double) {
    this(x, y, length, length)
  }

  def contains(point: Vector[Double]): Boolean = {
    require(point.length == 2)
    val xTest = point(0)
    val yTest = point(1)
    (x - hWidth <= xTest) && (x + hWidth >= xTest) && (y - hHeigth <= yTest) && (y + hHeigth >= yTest)
  }

  override def equals(other: Any): Boolean = {
    other match {
      case that: Cell =>
        this.x == that.x && this.y == that.y && this.hWidth == that.hWidth && this.hHeigth == that.hHeigth
    }
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1

    result = (prime * result + x).toInt
    result = (prime * result + y).toInt
    result = (prime * result + hWidth).toInt
    result = (prime * result + hHeigth).toInt
    result
  }
}

object Cell {

  def apply(x: Double, y: Double, hWidth: Double, hHeigth: Double): Cell = {
    new Cell(x, y, hWidth, hHeigth)
  }

  def apply(x: Double, y: Double, length: Double): Cell = {
    new Cell(x, y, length)
  }
}