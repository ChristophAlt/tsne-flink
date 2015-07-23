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

import org.apache.flink.ml.math.Vector


object ZOrder {
  // returns true if Vector a greater Vector b
  def compareByZorder(a: Vector, b: Vector): Boolean = {
    require(a.size == b.size, "The each size of vectors must be same to calculate distance.")
    var j = 0
    var x = 0L
    for (i <- 0 until a.size) {
      val y = java.lang.Double.doubleToRawLongBits(a(i)) ^ java.lang.Double.doubleToRawLongBits(b(i))
      if (less_msb(x, y)) {
        j = i
        x = y
      }
    }
    a(j) > b(j)
  }

  private def less_msb(x: Long, y: Long): Boolean = {
    (x < y) && (x < (x ^ y))
  }
}
