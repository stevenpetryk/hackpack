<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlightjs-line-numbers.js/1.1.0/highlightjs-line-numbers.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>
<script>hljs.initLineNumbersOnLoad();</script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto|Source+Code+Pro" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/styles/github.min.css" />
<link rel="stylesheet" href="main.css" />
<meta charset="utf-8" />

<!--
```java
import java.util.*;
```
-->

# General Programming

## Comparable

Sorting from smallest to largest:

<!--
```java
class Thing {
  int value;
``` -->

```java
// a.compareTo(b) < 0   =>   a < b   =>   a - b < 0
public int compareTo (Thing other) {
  return value - other.value;
}
```

<!--
```java
}
``` -->

<div class="page-break"></div>

# Permutation Generation
<div class="page-break"></div>

# Combination Generation
<div class="page-break"></div>

<!--
```java
class MathUtils {
```
-->

# GCD

```java
public static int gcd (int a, int b) {
  return b == 0 ? a : gcd(b, a%b);
}
```

# LCM

```java
public static int lcm (int a, int b) {
  return a * (b / gcd(a, b));
}
```

<div class="page-break"></div>

<!--
```java
}
```
-->

# Graphs

```java
class Node {
  public List<Node> children;
  public Node () { children = new ArrayList<Node>(); }

  public Node addChild (Node child) {
    children.add(child);
    child.children.add(this); // if undirected graph

    return this;
  }
}
```

## Kruskal's Algorithm
<div class="page-break"></div>

## Depth First Search
<div class="page-break"></div>

## Breadth First Search

```java
class BFS {
  public static int distanceToNode (Node start, Node target) {
    Set<Node> visited = new HashSet<>();
    Deque<NodeWithDistance> queue = new ArrayDeque<>();
    queue.add(new NodeWithDistance(start, 0));

    while (!queue.isEmpty()) {
      NodeWithDistance current = queue.poll();

      if (!visited.add(current.node)) continue;
      if (current.node == target) return current.distance;

      for (Node child : current.node.children) {
        queue.add(new NodeWithDistance(child, current.distance + 1));
      }
    }

    return -1;
  }

  static class NodeWithDistance {
    Node node; int distance;
    public NodeWithDistance (Node _node, int _distance) { node = _node; distance = _distance; }
  }
}
```

<div class="page-break"></div>

# Topological Sort
<div class="page-break"></div>

# Floyd-Warshall's Algorithm
<div class="page-break"></div>

# Dijkstra's Algorithm
<div class="page-break"></div>

# Bellman Ford's Algorithm
<div class="page-break"></div>

# Network Flow
<div class="page-break"></div>

# Matrix Chain Multiplication
<div class="page-break"></div>

# Longest Common Subsequence
<div class="page-break"></div>

# Knapsack DP

```java
class Knapsack {
  public static int knapsack (int capacity, int[] weights, int[] values, boolean allowDups) {
    int n = weights.length;
    int[] dp = new int[capacity + 1];

    for (int i = 0; i < n; i++) {
      for (
        int w = allowDups ? weights[i] : capacity;
        allowDups ? w <= capacity : w >= weights[i];
      ) {
        dp[w] = Math.max(dp[w], dp[w-weights[i]] + values[i] );

        if (allowDups) w++; else w--;
      }
    }

    return dp[capacity];
  }
}
```

<div class="page-break"></div>

# Line-Line Intersection
<div class="page-break"></div>

# Line-Plane Intersection
<div class="page-break"></div>

# Polygon Area
<div class="page-break"></div>

# Convex Hull
<div class="page-break"></div>

# Point in Polygon
<div class="page-break"></div>

# Tests

```java
public class hackpack {
  public static boolean failures = false;

  public static void main (String args[]) {
    testGCD();
    testLCM();
    testBFS();
    testKnapsack();

    if (!failures) {
      handleSuccess();
    }
  }

  public static void testGCD () {
    assertEqual(MathUtils.gcd(1, 1), 1);
    assertEqual(MathUtils.gcd(5, 10), 5);
    assertEqual(MathUtils.gcd(15, 3), 3);
  }

  public static void testLCM () {
    assertEqual(MathUtils.lcm(1, 1), 1);
    assertEqual(MathUtils.lcm(5, 10), 10);
    assertEqual(MathUtils.lcm(8, 3), 24);
  }

  public static void testBFS () {
    Node start = new Node();
    Node reachable = new Node();
    Node unreachable = new Node();

    start
      .addChild(new Node())
      .addChild(new Node())
      .addChild(
        new Node()
          .addChild(reachable)
          .addChild(new Node())
      );

    assertEqual(BFS.distanceToNode(start, reachable), 2);
    assertEqual(BFS.distanceToNode(start, unreachable), -1);
  }

  public static void testKnapsack () {
    int weights[] = new int[] { 3, 2, 6, 8, 1, 3 };
    int values[] = new int[] { 7, 5, 12, 20, 3, 6 };

    assertEqual(Knapsack.knapsack(1, weights, values, false), 3);
    assertEqual(Knapsack.knapsack(2, weights, values, false), 5);
    assertEqual(Knapsack.knapsack(10, weights, values, false), 25);
    assertEqual(Knapsack.knapsack(23, weights, values, false), 53);
  }

  /*
   * Low-level test code. Don't worry about this too much.
   */

  public static final String ANSI_RESET = "\u001B[0m";
  public static final String ANSI_RED = "\u001B[31m";
  public static final String ANSI_GREEN = "\u001B[32m";

  private static void handleSuccess () {
    System.out.println(ANSI_GREEN + "✓ All tests passed" + ANSI_RESET);
  }

  private static void handleTestFailure (TestFailure e) {
    failures = true;

    String failingTest = "";

    outer: for (StackTraceElement element : e.getStackTrace()) {
      String name = element.getMethodName();

      if (!name.startsWith("assert") && !name.startsWith("throwOn")) {
        failingTest = name;
        break outer;
      }
    }

    System.out.println(ANSI_RED + "× " + failingTest + " failed: " + e.getMessage() + ANSI_RESET);

    e.printStackTrace();
  }

  private static void assertEqual (int a, int b) {
    try {
      if (a != b) {
        throw new TestFailure(String.format("Expected %d to equal %d", a, b));
      }
    } catch (TestFailure e) {
      handleTestFailure(e);
    }
  }
}

class TestFailure extends Exception {
  public TestFailure (String message) {
    super(message);
  }
}
```
