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

package de.tu_berlin.dima.impro3;

import org.apache.flink.api.common.accumulators.Accumulator;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.HashMap;


public class MapAccumulator implements Accumulator<Tuple2<Integer, Double>, HashMap<Integer, Double>> {

    private static final long serialVersionUID = 1L;

    private HashMap<Integer, Double> localValue = new HashMap<Integer, Double>();

    @Override
    public void add(Tuple2<Integer, Double> kv) {
        Integer key = kv.f0;
        Double value = kv.f1;

        if (localValue.containsKey(key)) {
            localValue.put(key, localValue.get(key) + value);
        } else {
            localValue.put(key, value);
        }
    }

    @Override
    public HashMap<Integer, Double> getLocalValue() {
        return localValue;
    }

    @Override
    public void resetLocal() {
        localValue.clear();
    }

    @Override
    public void merge(Accumulator<Tuple2<Integer, Double>, HashMap<Integer, Double>> other) {
        HashMap<Integer, Double> lv = other.getLocalValue();
        for (Integer key : lv.keySet()) {
            if (localValue.containsKey(key)) {
                localValue.put(key, localValue.get(key) + lv.get(key));
            } else {
                localValue.put(key, lv.get(key));
            }
        }
    }

    @Override
    public Accumulator<Tuple2<Integer, Double>, HashMap<Integer, Double>> clone() {
        MapAccumulator newInstance = new MapAccumulator();
        newInstance.localValue = new HashMap<Integer, Double>(localValue);
        return newInstance;
    }

    @Override
    public String toString() {
        return "Map Accumulator " + localValue;
    }
}